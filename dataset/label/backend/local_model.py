"""Start the sibling Qwen 27B service only when the local port is free."""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LocalModelService:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.qwen_root = Path(__file__).resolve().parents[3].parent / "qwen"
        self.process: subprocess.Popen | None = None
        self.error: str | None = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(self, path: str) -> dict:
        with self.opener.open("http://127.0.0.1:8080" + path, timeout=1) as response:
            return json.load(response)

    def status(self) -> dict[str, str]:
        try:
            if self._get("/health").get("status") == "ok":
                models = self._get("/v1/models").get("data", [])
                if any(model.get("id") == "qwen-local"
                       and model.get("meta", {}).get("n_params", 0) >= 25_000_000_000
                       for model in models):
                    return {"state": "ready", "detail": "本地 Qwen 27B 已就绪"}
                return {"state": "occupied", "detail": "8080 端口运行的不是目标 27B 模型"}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.process is not None:
            if self.process.poll() is None:
                return {"state": "starting", "detail": "本地 Qwen 27B 正在加载"}
            return {"state": "error", "detail": self.error or f"模型服务退出，代码 {self.process.returncode}；查看 local_model.log"}
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=0.5):
                return {"state": "occupied", "detail": "8080 端口已被占用，无法启动本地模型"}
        except OSError:
            return {"state": "error" if self.error else "stopped", "detail": self.error or "本地模型尚未启动"}

    def ensure_running(self) -> dict[str, str]:
        current = self.status()
        if current["state"] != "stopped":
            return current
        executable = self.qwen_root / "third_party/llama.cpp/build-cuda/bin/llama-server"
        model = self.qwen_root / "models/Qwen3.8-27B-Q3_K_M.gguf"
        if not executable.is_file() or not model.is_file():
            self.error = "找不到隔壁 qwen 项目的 CUDA 服务或 27B 量化权重"
            return self.status()
        command = [
            str(executable), "--model", str(model), "--alias", "qwen-local",
            "--host", "127.0.0.1", "--port", "8080", "--n-gpu-layers", "all",
            "--ctx-size", "32768", "--parallel", "1", "--threads", "8",
            "--jinja", "--reasoning", "off", "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--ubatch-size", "128",
        ]
        try:
            with (self.data_root / "local_model.log").open("ab") as log:
                self.process = subprocess.Popen(
                    command, cwd=self.qwen_root, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            self.error = f"无法启动本地模型：{exc}"
        return self.status()

    def wait_ready(self, timeout_seconds: int = 90) -> None:
        self.ensure_running()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = self.status()
            if state["state"] == "ready":
                return
            if state["state"] in {"error", "occupied"}:
                raise RuntimeError(state["detail"])
            time.sleep(1)
        raise RuntimeError("本地模型加载超时；查看 local_model.log")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
