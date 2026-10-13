"""教务问答 API 启动入口。

用法：
    python run_api.py

默认监听 http://0.0.0.0:8000，Swagger 文档：http://localhost:8000/docs
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import uvicorn


def main():
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    print(f"启动教务问答 API：http://{host}:{port}")
    print(f"Swagger 文档：http://localhost:{port}/docs")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
