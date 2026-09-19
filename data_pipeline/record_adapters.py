"""Shared source-record adapters for pretraining and the curation importer."""
from __future__ import annotations

from typing import Any, Callable, Mapping


def first_text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def join_parts(*parts: str) -> str | None:
    texts = [
        text
        for part in parts
        if isinstance(part, str) and (text := part.strip())
    ]
    if not texts:
        return None
    return "\n\n".join(texts)


def adapt_plain_text(row: Mapping[str, Any]) -> str | None:
    return join_parts(
        first_text(row, "title"),
        first_text(row, "text", "content"),
    )


def adapt_classification_text(row: Mapping[str, Any]) -> str | None:
    return join_parts(first_text(row, "text", "content"))


def adapt_instruction_input_output(row: Mapping[str, Any]) -> str | None:
    instruction = first_text(row, "instruction", "prompt", "question", "query")
    input_text = first_text(row, "input", "context")
    output = first_text(row, "output", "answer", "response")
    return join_parts(instruction, input_text, output)


def adapt_question_answer(row: Mapping[str, Any]) -> str | None:
    question = first_text(row, "question", "query", "prompt")
    answer = first_text(row, "answer", "output", "response")
    return join_parts(question, answer)


def adapt_question_answer_optional_think(
    row: Mapping[str, Any],
) -> str | None:
    question = first_text(row, "question", "query", "prompt")
    think = first_text(row, "think", "reasoning")
    answer = first_text(row, "answer", "output", "response")
    return join_parts(question, think, answer)


def adapt_conversation(row: Mapping[str, Any]) -> str | None:
    conversation = row.get("conversations")
    if conversation is None:
        conversation = row.get("conversation")
    if conversation is None:
        conversation = row.get("messages")

    if isinstance(conversation, str):
        return join_parts(conversation)
    if not isinstance(conversation, list | tuple):
        return None

    contents = []
    for turn in conversation:
        if isinstance(turn, Mapping):
            content = first_text(turn, "content", "value", "text")
        elif isinstance(turn, str):
            content = turn.strip()
        else:
            content = ""
        if content:
            contents.append(content)
    return join_parts(*contents)


def adapt_tieba_thread(row: Mapping[str, Any]) -> str | None:
    title = first_text(row, "标题", "title")
    author_content = first_text(row, "楼主内容", "content", "text")
    raw_replies = row.get("回复列表")
    if raw_replies is None:
        raw_replies = row.get("replies")

    replies = []
    if isinstance(raw_replies, list | tuple):
        for reply in raw_replies:
            if isinstance(reply, Mapping):
                text = first_text(reply, "content", "value", "text")
            else:
                text = reply.strip() if isinstance(reply, str) else ""
            if text:
                replies.append(text)
    elif isinstance(raw_replies, str):
        text = raw_replies.strip()
        if text:
            replies.append(text)

    return join_parts(title, author_content, *replies)


ADAPTERS: dict[str, Callable[[Mapping[str, Any]], str | None]] = {
    "plain_text": adapt_plain_text,
    "classification_text": adapt_classification_text,
    "instruction_input_output": adapt_instruction_input_output,
    "question_answer": adapt_question_answer,
    "question_answer_optional_think": adapt_question_answer_optional_think,
    "sharegpt_conversations": adapt_conversation,
    "openai_role_content_conversation": adapt_conversation,
    "tieba_thread": adapt_tieba_thread,
}

ADAPTER_COLUMNS = {
    "plain_text": {"title", "text", "content"},
    "classification_text": {"text", "content"},
    "instruction_input_output": {
        "instruction",
        "prompt",
        "question",
        "query",
        "input",
        "context",
        "output",
        "answer",
        "response",
    },
    "question_answer": {
        "question",
        "query",
        "prompt",
        "answer",
        "output",
        "response",
    },
    "question_answer_optional_think": {
        "question",
        "query",
        "prompt",
        "think",
        "reasoning",
        "answer",
        "output",
        "response",
    },
    "sharegpt_conversations": {"conversations", "conversation", "messages"},
    "openai_role_content_conversation": {"conversation", "conversations", "messages"},
    "tieba_thread": {
        "标题",
        "title",
        "楼主内容",
        "content",
        "text",
        "回复列表",
        "replies",
    },
}


