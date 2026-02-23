import json

from loguru import logger
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware


def endpoint(path: str, method: str = 'GET'):
    def decorator(func):
        if not hasattr(func, '_route_settings'):
            func._route_settings = []
        func._route_settings.append((path, method.upper()))
        return func

    return decorator


class TypeWikiInstance(Starlette):
    def __init__(self, **kwargs):
        super().__init__(on_startup=[self.on_startup], on_shutdown=[self.on_shutdown], **kwargs)
        self._register_endpoints()
        self.setup_middlewares()
        self.setup_exception_handlers()

    def setup_middlewares(self):
        self.add_middleware(
            CORSMiddleware,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    def _register_endpoints(self):
        for attr in dir(self):
            attribute = getattr(self, attr)
            if callable(attribute) and hasattr(attribute, '_route_settings'):
                for path, method in attribute._route_settings:
                    self.add_route(path, attribute, methods=[method])

    def setup_exception_handlers(self):
        pass

    async def on_startup(self):
        """This method will be called on application start before any requests are processed."""
        pass

    async def on_shutdown(self):
        """This method will be called on application exit."""
        pass


# Custom JSON serializer for logs
def serialize(record):
    log_dict = {
        'timestamp': record['time'].strftime('%Y-%m-%d %H:%M:%S'),
        'level': record['level'].name,
        'message': record['message'],
        'function': record['function'],
        'line': record['line'],
        'file': record['file'].name,
    }
    return json.dumps(log_dict)


# Custom sink function that uses the serializer
def sink(message):
    serialized = serialize(message.record)
    print(serialized)


# Remove the default stderr logger
logger.remove()

# Add a new logger configuration to use the custom sink
logger.add(sink, level='INFO')
