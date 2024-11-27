import argparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from node import Node
from middleware import TransactionIDMiddleware
from common.exceptions import register_exception_handlers


def __init_cli() -> argparse:
    parser = argparse.ArgumentParser(
        description="OSS RAG App Server", usage="python main.py -l INFO -d ./plugins"
    )
    parser.add_argument(
        "-l",
        "--log",
        help="""
        Specify log level which should use. If you specify a value, it will be set to that value. If the value is missing, 
        the value is read from the configuration file(logging.level). If neither exists, it is set to DEBUG. Choose between the following options
        CRITICAL, ERROR, WARNING, INFO, DEBUG
        """,
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="""
        (Optional) Supply a directory where plugins should be loaded from. If you specify a value, it will be set to that value. If the value is missing, 
        the value is read from the configuration file(plugin_dir). If neither exists, it is set to ./plugins.
        """,
    )
    return parser


if __name__ == "__main__":
    __cli_args = __init_cli().parse_args()
    node = Node({"log_level": __cli_args.log, "directory": __cli_args.directory})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize the plugin manager object and load each plugin
        await node.initialize(app)
        yield
        # Clean up the plugin models and release the resources
        await node.finalize()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TransactionIDMiddleware)

    app.include_router(node.router)
    register_exception_handlers(app)
    uvicorn.run(
        app,
        host=node.config_data["http"]["host"],
        port=node.config_data["http"]["port"],
    )
