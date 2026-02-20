from typewiki.utils import TypeWikiInstance, logger


class TypeWikiApp(TypeWikiInstance):
    async def on_startup(self):
        logger.info('TypeWiki application starting!')

    async def on_shutdown(self):
        logger.info('Bon Voyage!')
