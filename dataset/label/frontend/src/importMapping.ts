export type Mapping = {
  text_fields: string[];
  text_separator: string;
  title_field: string | null;
  url_field: string | null;
  local_id_field: string | null;
  metadata_fields: string[];
  record_adapter: string | null;
};

export const recordFormats = [
  ["", "自定义正文列"],
  ["plain_text", "文章：标题 + 正文"],
  ["classification_text", "分类语料：仅正文"],
  ["instruction_input_output", "指令 + 输入 + 输出"],
  ["question_answer", "问题 + 回答"],
  ["question_answer_optional_think", "问题 + 思考 + 回答"],
  ["sharegpt_conversations", "ShareGPT 多轮对话"],
  ["openai_role_content_conversation", "role/content 多轮对话"],
  ["tieba_thread", "贴吧：标题 + 楼主 + 回复"],
] as const;

function findField(fields: string[], candidates: string[]): string | null {
  const byLowercase = new Map(fields.map((field) => [field.toLowerCase(), field]));
  for (const candidate of candidates) {
    const field = byLowercase.get(candidate.toLowerCase());
    if (field) return field;
  }
  return null;
}

export function inferMapping(fields: string[]): Mapping {
  const title = findField(fields, ["title", "标题", "name"]);
  const directText = findField(fields, ["text"]);
  const content = findField(fields, ["content", "body", "article", "document"]);
  const has = (name: string) => fields.includes(name);
  let recordAdapter: string | null = null;
  if (has("楼主内容") || has("回复列表") || has("replies")) recordAdapter = "tieba_thread";
  else if (has("messages") || has("conversation")) recordAdapter = "openai_role_content_conversation";
  else if (has("conversations")) recordAdapter = "sharegpt_conversations";
  else if (has("instruction")) recordAdapter = "instruction_input_output";
  else if (["question", "query", "prompt"].some(has) && ["answer", "output", "response"].some(has)) {
    recordAdapter = has("think") || has("reasoning") ? "question_answer_optional_think"
      : has("input") || has("context") ? "instruction_input_output" : "question_answer";
  }
  const dataType = findField(fields, ["dataType", "category", "source"]);
  return {
    text_fields: directText ? [directText] : content
      ? title && title !== content ? [title, content] : [content]
      : fields.length === 1 ? [fields[0]] : [],
    text_separator: "\n\n",
    title_field: title,
    url_field: findField(fields, ["url", "link", "source_url"]),
    local_id_field: findField(fields, ["uniqueKey", "doc_id", "document_id", "id"]),
    metadata_fields: dataType ? [dataType] : [],
    record_adapter: recordAdapter,
  };
}
