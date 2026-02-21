from pydantic import SecretStr
from pydantic_settings import BaseSettings


class TypeWikiConfig(BaseSettings):
    openai_model_name: str = 'gpt-5-mini'
    openai_api_key: SecretStr
