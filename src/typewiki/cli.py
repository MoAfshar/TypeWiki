import uvicorn


def main():
    uvicorn.run('typewiki.app:app', host='0.0.0.0', port=8000, log_level='error', reload=False)
