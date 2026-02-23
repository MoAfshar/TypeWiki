from json import JSONDecodeError

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from typewiki.config import TypeWikiConfig
from typewiki.datamodels import ChatRequest, ChatResponse
from typewiki.exceptions import PineconeIndexNotProvisionedError
from typewiki.prompts.copilot import build_prompt
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

        self.model = init_chat_model(self.config.openai_model_name, temperature=0.5)

        timeout = aiohttp.ClientTimeout(self.config.http_client_timeout_seconds)
        self.client = aiohttp.ClientSession(timeout=timeout)

        logger.info('Application configuration and setup is complete and running.')

    async def on_shutdown(self):
        await self.client.close()
        logger.info('Bon Voyage!')

    def setup_exception_handlers(self):
        self.add_exception_handler(JSONDecodeError, self._handle_json_decode_error)
        self.add_exception_handler(TypeError, self._handle_type_error)
        self.add_exception_handler(ValidationError, self._handle_validation_error)

    async def _handle_json_decode_error(self, _: Request, exc: Exception):
        logger.error(f'Invalid JSON in request body: {exc}')
        return JSONResponse(
            {'error': 'Invalid JSON in request body', 'detail': str(exc)},
            status_code=400,
        )

    async def _handle_type_error(self, _: Request, exc: Exception):
        logger.error(f'Invalid request body type: {exc}')
        return JSONResponse(
            {'error': 'Request body must be a JSON object', 'detail': str(exc)},
            status_code=400,
        )

    async def _handle_validation_error(self, _: Request, exc: Exception):
        logger.error(f'Request validation failed: {exc}')
        detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
        return JSONResponse(
            {'error': 'Request validation failed', 'detail': detail},
            status_code=422,
        )

    @endpoint(path='/health', method='GET')
    async def health(self, _: Request):
        return JSONResponse({'status': 'healthy'}, status_code=200)

    @endpoint(path='/v1/chat', method='POST')
    async def chat(self, request: Request):
        body = await request.json()
        data = ChatRequest(**body)

        article_context = await self.vector_store.asimilarity_search_with_score(data.message, k=5)

        logger.info(f'Processing user request with the message: {data.message}')

        prompt = build_prompt(
            user_message=data.message,
            search_results=article_context,
            history=data.history,
        )

        model_response = await self.model.ainvoke(prompt)

        response = ChatResponse(
            conversation_id=data.session_id,
            answer=model_response.content,
            model=self.config.openai_model_name,
        )

        logger.info(f'Successfully processed query for: {data.message}')
        return JSONResponse(response.model_dump(mode='json'), status_code=200)
