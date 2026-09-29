"""Share a fixed web chat session across documents while keeping their context separate."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path


DEEPSEEK_WEB_BATCH_SLOTS = 1  # Start conservatively; account-level feedback controls pacing.
QWEN_WEB_BATCH_SLOTS = 1
KIMI_WEB_BATCH_SLOTS = 1
DOUBAO_WEB_BATCH_SLOTS = 1
CHATGLM_WEB_BATCH_SLOTS = 1
SPARK_WEB_BATCH_SLOTS = 1
WENXIN_WEB_BATCH_SLOTS = 1
YUANBAO_WEB_BATCH_SLOTS = 1
WEB_SESSION_CLEANUP_SLOTS = 6  # Also remove sessions from older six-slot Qwen runs.
WEB_AUTH_SOURCES = {
    "qwen_web": ("QWEN_WEB_AUTH_FILE", ".qwen-web-auth.json", "https://chat2.qianwen.com", "Qwen-web"),
    "kimi_web": ("KIMI_WEB_AUTH_FILE", ".kimi-web-auth.json", "https://www.kimi.com", "Kimi-web"),
    "doubao_web": ("DOUBAO_WEB_AUTH_FILE", ".doubao-web-auth.json", "https://www.doubao.com", "Doubao-web"),
    "chatglm_web": ("CHATGLM_WEB_AUTH_FILE", ".chatglm-web-auth.json", "https://chatglm.cn", "GLM-Flash-web"),
    "spark_web": ("SPARK_WEB_AUTH_FILE", ".spark-web-auth.json", "https://spark.xfyun.cn", "Spark-Fast-web"),
    "wenxin_web": ("WENXIN_WEB_AUTH_FILE", ".wenxin-web-auth.json", "https://wenxin.baidu.com", "Wenxin-Fast-web"),
    "yuanbao_web": ("YUANBAO_WEB_AUTH_FILE", ".yuanbao-web-auth.json", "https://yuanbao.tencent.com", "Yuanbao-Fast-web"),
}


class DeepSeekWebError(RuntimeError):
    pass


class DeepSeekWebSessionError(DeepSeekWebError):
    pass


class DeepSeekWebHttpError(DeepSeekWebError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@lru_cache
def _node_executable() -> str:
    configured = os.environ.get("DEEPSEEK_WEB_NODE")
    if configured:
        return configured
    installed = shutil.which("node")
    if installed:
        return installed
    candidates = list((Path.home() / ".vscode-server/cli/servers").glob("Stable-*/server/node"))
    if candidates:
        return str(max(candidates, key=lambda path: path.stat().st_mtime))
    raise DeepSeekWebError("找不到 Node.js；请安装 Node 18+ 或设置 DEEPSEEK_WEB_NODE")


class WebChatClient:
    script_name = "web_chat_once.mjs"
    credential_env = "DEEPSEEK_WEB_TOKEN"
    label = "DeepSeek 网页"
    _clients: dict[tuple[type, Path, str], "WebChatClient"] = {}
    _clients_lock = threading.Lock()

    @classmethod
    def shared(cls, credential: str, timeout: float, state_path: Path):
        key = (cls, state_path.resolve(), hashlib.sha256(credential.encode()).hexdigest())
        with cls._clients_lock:
            if key not in cls._clients:
                cls._clients[key] = cls(credential, timeout, state_path)
            return cls._clients[key]

    def __init__(self, token: str, timeout: float, state_path: Path | None = None):
        self.token = token
        self.timeout = timeout
        self.script = Path(__file__).resolve().parents[3] / "deepseek-web-api" / self.script_name
        self.state_path = state_path
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._topic_id: str | None = None
        if state_path is not None and state_path.exists():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                if saved.get("token_sha256") == hashlib.sha256(token.encode()).hexdigest():
                    self._session_id = saved.get("session_id")
                    self._topic_id = saved.get("topic_id")
            except (OSError, ValueError):
                pass

    def _save_session(self, session_id: str, topic_id: str | None = None) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "token_sha256": hashlib.sha256(self.token.encode()).hexdigest(),
            "session_id": session_id,
            "topic_id": topic_id,
        }), encoding="utf-8")
        temporary.replace(self.state_path)

    def _remember_created_session(self, output: str | bytes | None) -> None:
        if not output:
            return
        try:
            first = json.loads(output.splitlines()[0])
            session_id = first["session_id"]
            topic_id = first.get("topic_id")
        except (ValueError, KeyError, TypeError, IndexError):
            return
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id
            self._topic_id = topic_id
            self._save_session(session_id, topic_id)

    def _command_argv(self, mode: str) -> list[str]:
        return [_node_executable(), str(self.script), mode]

    def _command(self, mode: str, payload: dict):
        try:
            result = subprocess.run(
                self._command_argv(mode),
                input=json.dumps(payload, ensure_ascii=False),
                text=True, capture_output=True, timeout=self.timeout,
                env={**os.environ, self.credential_env: self.token}, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._remember_created_session(exc.stdout)
            raise DeepSeekWebError(f"{self.label}请求超时") from exc
        except OSError as exc:
            raise DeepSeekWebError(f"无法启动{self.label}请求脚本") from exc
        if result.returncode:
            self._remember_created_session(result.stdout)
            detail = result.stderr.strip().replace(self.token, "[redacted]")[:500]
            if "invalid chat session id" in detail.lower():
                raise DeepSeekWebSessionError(f"{self.label}会话已失效")
            status = re.search(r"HTTP (\d{3})", detail)
            if status:
                raise DeepSeekWebHttpError(int(status.group(1)), f"{self.label}请求失败：{detail}")
            raise DeepSeekWebError(f"{self.label}请求失败：{detail or '脚本退出码 ' + str(result.returncode)}")
        try:
            return json.loads(result.stdout.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise DeepSeekWebError(f"{self.label}返回了无效响应") from exc

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        answer = self._command("--json-stdin", {
            "prompt": prompt, "session_id": session_id, "topic_id": topic_id,
            "parent_message_id": None,
        })
        content, used_session = answer.get("content"), answer.get("session_id")
        if not isinstance(content, str) or not content.strip() or not isinstance(used_session, str) or not used_session:
            raise DeepSeekWebError(f"{self.label}没有返回正文或会话 ID")
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content.strip()}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        return response, used_session, answer.get("topic_id")

    def complete(self, messages: list[dict]) -> dict:
        prompt = "\n\n".join(f"{message['role']}：\n{message['content']}" for message in messages)
        # A document may split into parallel chunks. Keep one request at a time
        # on each fixed conversation; distinct batch slots still run together.
        with self._lock:
            for attempt in range(2):
                if self._session_id is None:
                    response, self._session_id, self._topic_id = self._run(prompt, None, None)
                    self._save_session(self._session_id, self._topic_id)
                    return response
                session_id, topic_id = self._session_id, self._topic_id
                try:
                    response, self._session_id, self._topic_id = self._run(prompt, session_id, topic_id)
                    self._save_session(self._session_id, self._topic_id)
                    return response
                except DeepSeekWebSessionError:
                    if attempt:
                        raise
                    self._session_id = self._topic_id = None
                    if self.state_path is not None:
                        self.state_path.unlink(missing_ok=True)
        raise DeepSeekWebSessionError(f"{self.label}会话已失效")

    def delete(self) -> bool:
        """Delete only this client's saved web session; keep state if deletion fails."""
        with self._lock:
            if self._session_id is None:
                return False
            self._command("--delete-session-stdin", {"session_id": self._session_id})
            self._session_id = self._topic_id = None
            if self.state_path is not None:
                self.state_path.unlink(missing_ok=True)
            return True


class PacedWebChatClient(WebChatClient):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._provider_lock = threading.Lock()

    def complete(self, messages: list[dict]) -> dict:
        # Interactive and batch clients have separate session files, but one
        # website account must have only one in-flight chat across both.
        with type(self)._provider_lock:
            return super().complete(messages)

    def delete(self) -> bool:
        with type(self)._provider_lock:
            return super().delete()

    _pace_lock = threading.Lock()
    _next_start = 0.0
    _request_gap = 15.0
    _min_gap = 15.0
    _max_gap = 300.0
    _successes = 0

    def _mark_rate_limited(self) -> None:
        cls = type(self)
        with cls._pace_lock:
            if cls._request_gap <= 0:  # Unit tests may disable the clock.
                return
            cls._request_gap = min(cls._max_gap, max(cls._min_gap, cls._request_gap * 2))
            cls._next_start = max(cls._next_start, time.monotonic() + cls._request_gap)
            cls._successes = 0

    def _mark_success(self) -> None:
        cls = type(self)
        with cls._pace_lock:
            if cls._request_gap <= 0:
                return
            cls._successes += 1
            if cls._successes >= 12:
                cls._request_gap = max(cls._min_gap, cls._request_gap * 0.9)
                cls._successes = 0

    def _finish_turn(self) -> None:
        cls = type(self)
        with cls._pace_lock:
            cls._next_start = max(cls._next_start, time.monotonic() + cls._request_gap)

    def _wait_turn(self) -> None:
        cls = type(self)
        while True:
            with cls._pace_lock:
                now = time.monotonic()
                delay = cls._next_start - now
                if delay <= 0:
                    cls._next_start = now + cls._request_gap
                    return
            time.sleep(min(delay, 5))


class DeepSeekWebClient(PacedWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    _request_gap = 20.0
    _min_gap = 20.0
    _successes = 0

    def _defer(self, seconds: float) -> None:
        cls = type(self)
        with cls._pace_lock:
            cls._next_start = max(cls._next_start, time.monotonic() + seconds)

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        for attempt in range(2):
            self._wait_turn()
            try:
                result = super()._run(prompt, session_id, topic_id)
                self._mark_success()
                return result
            except DeepSeekWebError as exc:
                message = str(exc)
                if "user is muted" in message:
                    raise
                if "parallel_chat_limit" in message:
                    self._mark_rate_limited()
                    self._defer(4)
                elif ("rate_limit_reached" in message
                      or isinstance(exc, DeepSeekWebHttpError) and exc.status == 429):
                    self._mark_rate_limited()
                    self._defer(15)
                else:
                    raise
                if attempt:
                    raise
                session_id = self._session_id or session_id
                topic_id = self._topic_id or topic_id
            finally:
                self._finish_turn()
        raise DeepSeekWebError("DeepSeek 网页请求失败")


class QwenWebClient(PacedWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    _request_gap = 20.0
    _min_gap = 20.0
    _successes = 0
    script_name = "qwen_chat.mjs"
    credential_env = "QWEN_WEB_AUTH_FILE"
    label = "千问网页"

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        self._wait_turn()
        try:
            result = super()._run(prompt, session_id, topic_id)
        except DeepSeekWebHttpError as exc:
            if exc.status == 429:
                self._mark_rate_limited()
            raise
        else:
            self._mark_success()
            return result
        finally:
            self._finish_turn()


class KimiWebClient(PacedWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    _request_gap = 15.0
    _min_gap = 15.0
    _successes = 0
    script_name = "kimi_chat.mjs"
    credential_env = "KIMI_WEB_AUTH_FILE"
    label = "Kimi 网页"

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        self._wait_turn()
        try:
            result = super()._run(prompt, session_id, topic_id)
        except DeepSeekWebHttpError as exc:
            if exc.status == 429:
                self._mark_rate_limited()
            raise
        else:
            self._mark_success()
            return result
        finally:
            self._finish_turn()


class DoubaoWebClient(PacedWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    _request_gap = 15.0
    _min_gap = 15.0
    _successes = 0
    script_name = "doubao_chat.py"
    credential_env = "DOUBAO_WEB_AUTH_FILE"
    label = "豆包网页"

    def _command_argv(self, mode: str) -> list[str]:
        return [sys.executable, str(self.script), mode]

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        self._wait_turn()
        try:
            result = super()._run(prompt, session_id, topic_id)
        except DeepSeekWebError as exc:
            if ("没有返回 SSE 聊天流" in str(exc)
                    or "rate limited" in str(exc).lower()
                    or isinstance(exc, DeepSeekWebHttpError) and exc.status == 429):
                self._mark_rate_limited()
            raise
        else:
            self._mark_success()
            return result
        finally:
            self._finish_turn()


class PythonWebChatClient(PacedWebChatClient):
    script_name = "python_web_bridge.py"
    provider_name = ""
    _request_gap = 15.0
    _min_gap = 15.0

    def _command_argv(self, mode: str) -> list[str]:
        return [sys.executable, str(self.script), self.provider_name, mode]

    def _run(self, prompt: str, session_id: str | None, topic_id: str | None):
        self._wait_turn()
        try:
            result = super()._run(prompt, session_id, topic_id)
        except DeepSeekWebHttpError as exc:
            if exc.status == 429:
                self._mark_rate_limited()
            raise
        else:
            self._mark_success()
            return result
        finally:
            self._finish_turn()


class ChatglmWebClient(PythonWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    provider_name = "chatglm_web"
    credential_env = "CHATGLM_WEB_AUTH_FILE"
    label = "智谱清言网页"


class SparkWebClient(PythonWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    provider_name = "spark_web"
    credential_env = "SPARK_WEB_AUTH_FILE"
    label = "讯飞星火网页"


class WenxinWebClient(PythonWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    provider_name = "wenxin_web"
    credential_env = "WENXIN_WEB_AUTH_FILE"
    label = "文心网页"


class YuanbaoWebClient(PythonWebChatClient):
    _pace_lock = threading.Lock()
    _next_start = 0.0
    provider_name = "yuanbao_web"
    credential_env = "YUANBAO_WEB_AUTH_FILE"
    label = "腾讯元宝网页"


WEB_CLIENTS = {
    "deepseek_web": DeepSeekWebClient,
    "qwen_web": QwenWebClient,
    "kimi_web": KimiWebClient,
    "doubao_web": DoubaoWebClient,
    "chatglm_web": ChatglmWebClient,
    "spark_web": SparkWebClient,
    "wenxin_web": WenxinWebClient,
    "yuanbao_web": YuanbaoWebClient,
}


def cleanup_batch_sessions(root: Path, configs) -> list[str]:
    """Delete every fixed web session for the completed batch run."""
    errors = []
    for config in configs.values():
        if config.provider not in WEB_CLIENTS or not config.api_key:
            continue
        client_type = WEB_CLIENTS[config.provider]
        for slot in range(WEB_SESSION_CLEANUP_SLOTS):
            path = root / f"{config.provider}_batch_{slot}_session.json"
            if not path.exists():
                continue
            try:
                client_type.shared(config.api_key, config.timeout_seconds, path).delete()
            except Exception as exc:
                errors.append(f"{config.provider} 槽 {slot}：{exc}")
    return errors
