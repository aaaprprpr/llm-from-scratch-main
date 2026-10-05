import argparse

import uvicorn


parser = argparse.ArgumentParser(description="启动独立配置面板")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8010)
args = parser.parse_args()
uvicorn.run("config_panel.server:app", host=args.host, port=args.port)
