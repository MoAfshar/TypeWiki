from pydantic import SecretStr
from pydantic_settings import BaseSettings


class TypeWikiConfig(BaseSettings):
    openai_model_name: str = 'gpt-5-mini'
    openai_embedding_model_name: str = 'text-embedding-3-large'
    openai_api_key: SecretStr
    pinecone_api_key: SecretStr
    pinecone_index_name: str = 'typewiki-helpcenter-dev-v1'
