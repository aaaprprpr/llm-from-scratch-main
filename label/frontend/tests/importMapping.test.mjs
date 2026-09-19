import assert from "node:assert/strict";
import test from "node:test";
import { inferMapping } from "../src/importMapping.ts";

test("keeps existing wiki mapping and excludes classification labels", () => {
  assert.deepEqual(inferMapping(["id", "title", "text", "url"]).text_fields, ["text"]);
  assert.equal(inferMapping(["text", "label"]).record_adapter, null);
});

test("recognizes full structured records instead of one nested turn", () => {
  for (const [fields, expected] of [
    [["instruction", "input", "output"], "instruction_input_output"],
    [["question", "answer"], "question_answer"],
    [["query", "response", "reasoning"], "question_answer_optional_think"],
    [["prompt", "context", "response"], "instruction_input_output"],
    [["conversations", "conversations.0.value"], "sharegpt_conversations"],
    [["messages", "messages.0.content"], "openai_role_content_conversation"],
    [["标题", "楼主内容", "回复列表"], "tieba_thread"],
  ]) assert.equal(inferMapping(fields).record_adapter, expected);
});
