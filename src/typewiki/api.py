import aiohttp
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from typewiki.config import TypeWikiConfig
from typewiki.exceptions import PineconeIndexNotProvisionedError
from typewiki.utils import TypeWikiInstance, logger


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
        logger.info('Bon Voyage!')
