from typing import TypedDict

from chat_schema import ChatMessage


class PreferenceExample(TypedDict):
    chosen_messages: list[ChatMessage]
    rejected_messages: list[ChatMessage]
