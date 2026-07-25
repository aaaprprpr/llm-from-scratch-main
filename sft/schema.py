from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class ChatExample(TypedDict):
    messages: list[ChatMessage]


def make_message(role: Role, content: str) -> ChatMessage:
    content = content.strip()
    if not content:
        raise ValueError(f"{role} 消息内容不能为空")
    return {"role": role, "content": content}
