"""Structured stdin/stdout adapter for the four Python web chat examples."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = {
    "chatglm_web": "chatglm_chat.py",
    "spark_web": "spark_chat.py",
    "wenxin_web": "wenxin_chat.py",
    "yuanbao_web": "yuanbao_chat.py",
}


def load_module(provider: str):
    filename = MODULES[provider]
    spec = importlib.util.spec_from_file_location(provider, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def continue_state(session_id: str | None, topic_id: str | None) -> dict | None:
    if not session_id:
        return None
    if not topic_id:
        return {"session_id": session_id}
    state = json.loads(topic_id)
    if not isinstance(state, dict) or state.get("session_id") != session_id:
        raise RuntimeError("网页会话状态不匹配")
    return state


def send(provider: str, module, value, prompt: str,
         session_id: str | None, topic_id: str | None) -> dict:
    prior = continue_state(session_id, topic_id)
    chat_id = session_id
    if provider == "spark_web" and not chat_id:
        chat_id = module.create_chat(value)
        print(json.dumps({"session_id": chat_id}), flush=True)
    elif provider == "yuanbao_web" and not chat_id:
        chat_id = module.create_chat(value)
        print(json.dumps({"session_id": chat_id}), flush=True)
    answer = io.StringIO()
    with redirect_stdout(answer):
        if provider == "chatglm_web":
            state = module.chat(value, prompt, chat_id or "")
            chat_id = str(state["conversation_id"])
        elif provider == "spark_web":
            state = module.chat(value, prompt, chat_id, prior.get("sid") if prior else None)
            chat_id = str(state["chat_id"])
        elif provider == "wenxin_web":
            previous = ({"session_id": chat_id, "rank": prior["rank"]} if prior else None)
            state = module.chat(value, prompt, previous)
            chat_id = str(state["session_id"])
        else:
            state = module.chat(value, prompt, chat_id)
            chat_id = str(state["conversation_id"])
    content = answer.getvalue().strip()
    if not chat_id or not content:
        raise RuntimeError("网页聊天缺少会话 ID 或正文")
    topic = {"session_id": chat_id}
    if provider == "spark_web":
        topic["sid"] = state["sid"]
    elif provider == "wenxin_web":
        topic["rank"] = state["rank"]
    return {"session_id": chat_id, "topic_id": json.dumps(topic, ensure_ascii=False),
            "content": content}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in MODULES or sys.argv[2] not in {"--json-stdin", "--delete-session-stdin"}:
        raise RuntimeError("用法：python python_web_bridge.py <来源> <--json-stdin|--delete-session-stdin>")
    provider, mode = sys.argv[1:]
    request = json.load(sys.stdin)
    module = load_module(provider)
    value = module.auth()
    if mode == "--delete-session-stdin":
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("缺少网页会话 ID")
        module.delete_chat(value, session_id)
        print(json.dumps({"deleted": session_id}))
        return
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("缺少提示词")
    result = send(provider, module, value, prompt,
                  request.get("session_id"), request.get("topic_id"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
