"""Airflow DAG for ingesting Help Center PDF articles into vector storage.

This pipeline loads local PDF files, extracts and cleans text content,
splits it into appropriately-sized chunks, and prepares them with metadata
suitable for vector database storage (e.g., Pinecone).

The pipeline uses a manifest file (pdf_manifest.json) to map PDF filenames
to their source URLs and article titles for proper attribution.
"""

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.sdk import dag, task
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field, computed_field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PDF_DIR = Path(os.getenv('TYPEWIKI_PDF_DIR', PROJECT_ROOT / 'pdfs')).resolve()
MANIFEST_PATH = PDF_DIR / 'pdf_manifest.json'

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'typewiki-helpcenter-dev-v1')
PINECONE_DIMENSION = 3072  # text-embedding-3-large dimension
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL_NAME', 'text-embedding-3-large')
PINECONE_BATCH_SIZE = 100  # Pinecone recommends batches of 100


class ArticleMetadata(BaseModel):
    """Metadata for a single Help Center article PDF.

    Attributes:
        filename: Name of the PDF file (must match actual file in PDF_DIR).
        url: Original Help Center URL for this article.
        title: Human-readable article title.
        category: Optional category/section for grouping articles.
        last_updated: Optional ISO date when article was last updated.
    """

    filename: str = Field(..., description="PDF filename (e.g., 'multi-question-page.pdf')")
    url: str = Field(..., description='Original Help Center article URL')
    title: str = Field(..., description='Article title for display')
    category: str | None = Field(None, description='Article category/section')
    last_updated: str | None = Field(None, description='ISO date of last update')


class PDFManifest(BaseModel):
    """Manifest file containing metadata for all Help Center PDFs.

    Attributes:
        articles: List of article metadata entries.
    """

    articles: list[ArticleMetadata] = Field(default_factory=list)

    def get_by_filename(self, filename: str) -> ArticleMetadata | None:
        """Look up article metadata by PDF filename.

        Args:
            filename: The PDF filename to look up.

        Returns:
            ArticleMetadata if found, None otherwise.
        """
        for article in self.articles:
            if article.filename == filename:
                return article
        return None


class DocumentChunk(BaseModel):
    """Schema for a single document chunk ready for vector storage.

    This schema captures all metadata needed for effective retrieval and
    source attribution in a RAG system.

    Attributes:
        text: The cleaned text content of this chunk.
        source_path: Absolute path to the source PDF file.
        filename: Name of the source PDF file.
        source_url: Original Help Center URL for attribution.
        article_title: Human-readable title of the source article.
        article_category: Category/section of the article (optional).
        page_number: 1-indexed page number where this chunk originated.
        chunk_index: 0-indexed position of this chunk within the document.
        total_chunks: Total number of chunks from the source document.
        start_char_index: Character offset where this chunk starts in the
            original page content.
        chunk_size: Number of characters in this chunk.
        ingested_at: ISO timestamp when this chunk was processed.
    """

    text: str = Field(..., description='Cleaned text content of the chunk')
    source_path: str = Field(..., description='Absolute path to source PDF')
    filename: str = Field(..., description='Source PDF filename')
    source_url: str = Field(..., description='Original Help Center URL')
    article_title: str = Field(..., description='Article title for display')
    article_category: str | None = Field(None, description='Article category/section')
    page_number: int = Field(..., ge=1, description='1-indexed page number')
    chunk_index: int = Field(..., ge=0, description='0-indexed chunk position in document')
    total_chunks: int = Field(..., ge=1, description='Total chunks from source document')
    start_char_index: int = Field(..., ge=0, description='Character offset in original page')
    chunk_size: int = Field(..., ge=0, description='Number of characters in chunk')
    ingested_at: str = Field(..., description='ISO timestamp of ingestion')

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_id(self) -> str:
        """Generate a deterministic unique ID for this chunk.

        Returns:
            A SHA-256 hash based on source URL and chunk index. Using URL
            instead of path ensures consistent IDs across different machines.
        """
        unique_str = f"{self.source_url}:chunk{self.chunk_index}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    def to_pinecone_format(self) -> dict[str, Any]:
        """Convert chunk to Pinecone upsert format.

        Returns:
            Dict with 'id', 'metadata', and placeholder for 'values' (embeddings).
            The 'values' field should be populated with embeddings before upsert.
        """
        return {
            'id': self.chunk_id,
            'metadata': {
                'text': self.text,
                'source_url': self.source_url,
                'article_title': self.article_title,
                'article_category': self.article_category or '',
                'filename': self.filename,
                'page_number': self.page_number,
                'chunk_index': self.chunk_index,
                'total_chunks': self.total_chunks,
                'ingested_at': self.ingested_at,
            },
            'values': [],  # Placeholder for embedding vector
        }


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> PDFManifest:
    """Load the PDF manifest file containing article metadata.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        PDFManifest object with article metadata.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        ValueError: If manifest JSON is invalid.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}\n"
            f"Create a pdf_manifest.json file in {manifest_path.parent}"
        )

    with open(manifest_path) as f:
        data = json.load(f)

    return PDFManifest.model_validate(data)


def list_pdf_files(pdf_dir: Path) -> list[Path]:
    """List all PDF files in the specified directory.

    Args:
        pdf_dir: Path to directory containing PDF files.

    Returns:
        Sorted list of Path objects for each PDF file found.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    return sorted(pdf_dir.glob('*.pdf'))


def load_pdf_pages(pdf_path: Path, article_meta: ArticleMetadata) -> list[dict[str, Any]]:
    """Load all pages from a PDF file with metadata.

    Args:
        pdf_path: Path to the PDF file to load.
        article_meta: Metadata for this article from the manifest.

    Returns:
        List of dicts containing 'page_content', 'page_number', and 'metadata'
        for each page in the PDF.
    """
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    pages = []
    for i, doc in enumerate(docs):
        pages.append(
            {
                'page_content': doc.page_content,
                'page_number': i + 1,
                'metadata': {
                    'source_path': str(pdf_path),
                    'filename': pdf_path.name,
                    'source_url': article_meta.url,
                    'article_title': article_meta.title,
                    'article_category': article_meta.category,
                },
            }
        )
    return pages


def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text.

    Performs the following transformations:
    - Normalizes whitespace (multiple spaces/newlines to single)
    - Removes page artifacts like headers/footers with page numbers
    - Strips leading/trailing whitespace

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned and normalized text string.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a configured text splitter for document chunking.

    Returns:
        RecursiveCharacterTextSplitter configured with appropriate chunk size
        and overlap for RAG applications. Uses common separators optimized
        for help center article content.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=['\n\n', '\n', '. ', ', ', ' ', ''],
        length_function=len,
    )


def init_pinecone_index() -> Any:
    """Initialize Pinecone and create index if it doesn't exist.

    Returns:
        Pinecone Index object ready for upserts.

    Raises:
        ValueError: If PINECONE_API_KEY environment variable is not set.
    """
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError('PINECONE_API_KEY environment variable is required')

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully")
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)


def get_embeddings_client() -> OpenAIEmbeddings:
    """Create an OpenAI embeddings client.

    Returns:
        OpenAIEmbeddings instance configured with the embedding model.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable is required')

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key)


def chunk_document(
    pages: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    ingestion_timestamp: str,
) -> list[DocumentChunk]:
    """Split document pages into chunks with full metadata.

    Args:
        pages: List of page dicts from load_pdf_pages().
        splitter: Configured text splitter instance.
        ingestion_timestamp: ISO timestamp for this ingestion run.

    Returns:
        List of DocumentChunk objects ready for vector storage.
    """
    all_chunks: list[DocumentChunk] = []

    combined_text = ''
    page_boundaries: list[tuple[int, int, int]] = []

    for page in pages:
        cleaned = clean_text(page['page_content'])
        if not cleaned:
            continue

        start_pos = len(combined_text)
        combined_text += cleaned + '\n\n'
        end_pos = len(combined_text)
        page_boundaries.append((start_pos, end_pos, page['page_number']))

    if not combined_text.strip():
        return all_chunks

    text_chunks = splitter.split_text(combined_text)

    def find_page_for_position(pos: int) -> int:
        """Find which page a character position belongs to."""
        for start, end, page_num in page_boundaries:
            if start <= pos < end:
                return page_num
        return page_boundaries[-1][2] if page_boundaries else 1

    meta = pages[0]['metadata']
    source_path = meta['source_path']
    filename = meta['filename']
    source_url = meta['source_url']
    article_title = meta['article_title']
    article_category = meta.get('article_category')

    current_pos = 0
    for i, chunk_text in enumerate(text_chunks):
        chunk_start = combined_text.find(chunk_text, current_pos)
        if chunk_start == -1:
            chunk_start = current_pos

        page_number = find_page_for_position(chunk_start)

        chunk = DocumentChunk(
            text=chunk_text,
            source_path=source_path,
            filename=filename,
            source_url=source_url,
            article_title=article_title,
            article_category=article_category,
            page_number=page_number,
            chunk_index=i,
            total_chunks=len(text_chunks),
            start_char_index=chunk_start,
            chunk_size=len(chunk_text),
            ingested_at=ingestion_timestamp,
        )
        all_chunks.append(chunk)
        current_pos = chunk_start + 1

    return all_chunks


@dag(
    dag_id='typewiki_helpcenter_ingest',
    description='Ingest local Help Center PDFs into vector storage for RAG',
    start_date=datetime(2026, 2, 21),
    schedule='@daily',
    default_args={'retries': 2, 'retry_delay': timedelta(minutes=15)},
    catchup=False,
    tags=['typewiki', 'rag', 'ingest', 'helpcenter'],
)
def typewiki_helpcenter_ingest():
    """DAG for ingesting Help Center PDF articles.

    Pipeline stages:
        1. load_manifest_task: Load article metadata from manifest file
        2. extract_and_chunk: Load, clean, and chunk each PDF with metadata
        3. prepare_for_storage: Combine all chunks with final metadata
    """

    @task
    def load_manifest_task() -> list[dict[str, Any]]:
        """Load manifest and return list of articles to process.

        Returns:
            List of article metadata dicts with 'pdf_path' added.

        Raises:
            FileNotFoundError: If manifest or any referenced PDF is missing.
        """
        manifest = load_manifest()
        print(f"Loaded manifest with {len(manifest.articles)} articles")

        articles_to_process = []
        for article in manifest.articles:
            pdf_path = PDF_DIR / article.filename
            if not pdf_path.exists():
                print(f"  WARNING: PDF not found: {article.filename}")
                continue

            print(f"  - {article.title} ({article.filename})")
            article_dict = article.model_dump()
            article_dict['pdf_path'] = str(pdf_path)
            articles_to_process.append(article_dict)

        if not articles_to_process:
            raise FileNotFoundError('No valid PDFs found in manifest')

        return articles_to_process

    @task(show_return_value_in_logs=False)
    def extract_and_chunk(article_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract text from a PDF and split into chunks.

        Args:
            article_data: Article metadata dict with 'pdf_path' field.

        Returns:
            List of chunk dictionaries (serialized DocumentChunk models).
        """
        pdf_path = Path(article_data['pdf_path'])
        article_meta = ArticleMetadata.model_validate(article_data)

        print(f"Processing: {article_meta.title}")
        print(f"  URL: {article_meta.url}")

        pages = load_pdf_pages(pdf_path, article_meta)
        print(f"  Loaded {len(pages)} pages")

        splitter = create_text_splitter()
        timestamp = datetime.now(UTC).isoformat() + 'Z'

        chunks = chunk_document(pages, splitter, timestamp)
        print(f"  Created {len(chunks)} chunks")

        if chunks:
            print(
                f"  Chunk sizes: min={min(c.chunk_size for c in chunks)}, "
                f"max={max(c.chunk_size for c in chunks)}, "
                f"avg={sum(c.chunk_size for c in chunks) // len(chunks)}"
            )

        return [chunk.model_dump() for chunk in chunks]

    @task(show_return_value_in_logs=False)
    def prepare_for_storage(all_chunks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Combine and validate all chunks for storage.

        Args:
            all_chunks: List of chunk lists from each PDF.

        Returns:
            Summary dict with 'total_chunks', 'chunks' list, and 'stats'.
        """
        flattened = [
            DocumentChunk.model_validate(chunk) for chunk_list in all_chunks for chunk in chunk_list
        ]

        print(f"\nTotal chunks prepared: {len(flattened)}")

        by_source: dict[str, int] = {}
        for chunk in flattened:
            by_source[chunk.article_title] = by_source.get(chunk.article_title, 0) + 1

        print('Chunks per article:')
        for title, count in sorted(by_source.items()):
            print(f"  {title}: {count} chunks")

        return {
            'total_chunks': len(flattened),
            'chunks': [c.model_dump() for c in flattened],
            'stats': {
                'sources': len(by_source),
                'chunks_per_source': by_source,
            },
        }

    @task
    def upsert_to_pinecone(storage_data: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings and upsert chunks to Pinecone.

        This task:
        1. Initializes Pinecone and creates the index if needed
        2. Generates embeddings for all chunk texts using OpenAI
        3. Upserts vectors with metadata to Pinecone in batches

        Args:
            storage_data: Output from prepare_for_storage with 'chunks' list.

        Returns:
            Summary dict with upsert statistics.
        """
        chunks_data = storage_data['chunks']
        if not chunks_data:
            print('No chunks to upsert')
            return {'upserted': 0, 'status': 'empty'}

        print(f"\nInitializing Pinecone index: {PINECONE_INDEX_NAME}")
        index = init_pinecone_index()

        print(f"Initializing OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
        embeddings_client = get_embeddings_client()

        chunks = [DocumentChunk.model_validate(c) for c in chunks_data]
        texts = [chunk.text for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embeddings_client.embed_documents(texts)
        print(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

        vectors_to_upsert = []
        for chunk, embedding in zip(chunks, embeddings):
            pinecone_record = chunk.to_pinecone_format()
            pinecone_record['values'] = embedding
            vectors_to_upsert.append(pinecone_record)

        total_upserted = 0
        num_batches = (len(vectors_to_upsert) + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE

        print(f"Upserting {len(vectors_to_upsert)} vectors in {num_batches} batches...")

        for i in range(0, len(vectors_to_upsert), PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[i : i + PINECONE_BATCH_SIZE]
            batch_num = (i // PINECONE_BATCH_SIZE) + 1

            index.upsert(vectors=batch)
            total_upserted += len(batch)

            print(f"  Batch {batch_num}/{num_batches}: upserted {len(batch)} vectors")

        print(f"\nSuccessfully upserted {total_upserted} vectors to '{PINECONE_INDEX_NAME}'")

        index_stats = index.describe_index_stats()
        print(f"Index stats: {index_stats.total_vector_count} total vectors")

        return {
            'upserted': total_upserted,
            'index_name': PINECONE_INDEX_NAME,
            'embedding_model': OPENAI_EMBEDDING_MODEL,
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'status': 'success',
        }

    articles = load_manifest_task()
    chunks_per_pdf = extract_and_chunk.expand(article_data=articles)
    storage_ready = prepare_for_storage(chunks_per_pdf)
    upsert_to_pinecone(storage_ready)


typewiki_helpcenter_dag = typewiki_helpcenter_ingest()
