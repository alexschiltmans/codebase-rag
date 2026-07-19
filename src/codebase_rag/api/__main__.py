"""Entry point: `python -m codebase_rag.api` runs the API with uvicorn."""

import uvicorn

from codebase_rag.config import Config


def main() -> None:
    config = Config.get_instance()
    uvicorn.run(
        "codebase_rag.api.app:create_app",
        factory=True,
        host=config.api_host,
        port=config.api_port,
    )


if __name__ == "__main__":
    main()
