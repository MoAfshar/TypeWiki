import aiohttp
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from starlette.requests import Request
from starlette.responses import JSONResponse

from typewiki.config import TypeWikiConfig
from typewiki.datamodels import ChatRequest, ChatResponse
from typewiki.exceptions import PineconeIndexNotProvisionedError
from typewiki.utils import TypeWikiInstance, endpoint, logger


class TypeWikiApp(TypeWikiInstance):
    async def on_startup(self):
        logger.info('TypeWiki application starting!')
        self.config = TypeWikiConfig()

        pinecone = Pinecone(api_key=self.config.pinecone_api_key.get_secret_value())
        if not pinecone.has_index(self.config.pinecone_index_name):
            raise PineconeIndexNotProvisionedError(
                f"Pinecone index '{self.config.pinecone_index_name}' not found. "
                'Expected it to be created by the MLflow provisioning job.'
            )

        index = pinecone.Index(self.config.pinecone_index_name)
        embeddings = OpenAIEmbeddings(model=self.config.openai_embedding_model_name)
        self.vector_store = PineconeVectorStore(index=index, embedding=embeddings)

        timeout = aiohttp.ClientTimeout(self.config.http_client_timeout_seconds)
        self.client = aiohttp.ClientSession(timeout=timeout)

        logger.info('Application configuration and setup is complete and running.')

    async def on_shutdown(self):
        await self.client.close()
        logger.info('Bon Voyage!')

    @endpoint(path='/v1/chat', method='POST')
    async def chat(self, request: Request):
        data = ChatRequest(**(await request.json()))

        resp = ChatResponse(
            conversation_id=data.session_id,  # or rename to session_id everywhere for consistency
            answer='Stub answer (wire retrieval + LLM next).',
            sources=[],
            model=None,
        )

        return JSONResponse(resp.model_dump(mode='json'), status_code=200)
