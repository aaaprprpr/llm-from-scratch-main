import assert from "node:assert/strict";
import test from "node:test";
import { buildTextDiff } from "../src/textDiff.ts";

function assertRoundTrip(original, draft) {
  const diff = buildTextDiff(original, draft);
  assert.equal(diff.parts.filter(part => part.kind !== "added").map(part => part.value).join(""), original);
  assert.equal(diff.parts.filter(part => part.kind !== "removed").map(part => part.value).join(""), draft);
  return diff;
}

test("partial removal marks the selected garbage, preserving Chinese body", () => {
  const diff = assertRoundTrip("正文开始。点击广告领取优惠！正文继续。", "正文开始。正文继续。");
  assert.equal(diff.parts.filter(part => part.kind === "removed").map(part => part.value).join(""), "点击广告领取优惠！");
  assert.equal(diff.added, 0);
});

test("edits and traditional to simplified conversion show both old and new characters", () => {
  const diff = assertRoundTrip("數學🙂很有趣", "数学🙃非常有趣");
  assert.ok(diff.removed > 0 && diff.added > 0);
  for (const part of diff.parts) assert.ok(part.value.isWellFormed());
});

test("moving fragments across paragraphs preserves both recoverable versions", () => {
  const diff = assertRoundTrip("第一段主体。移到后面这句话。\n\n第二段主体。", "第一段主体。\n\n第二段主体。移到后面这句话。");
  assert.ok(diff.removed > 0 && diff.added > 0);
});

test("merging paragraphs, full deletion and undo all stay visible", () => {
  const original = "前半句\n\n后半句。";
  assert.equal(assertRoundTrip(original, "前半句后半句。").removed, 2);
  assert.equal(assertRoundTrip(original, "").added, 0);
  const undone = assertRoundTrip(original, original);
  assert.equal(undone.removed + undone.added, 0);
});

test("large rewrites fall back without dropping text from either version", () => {
  const original = "原始材料甲乙丙\n".repeat(10000);
  const draft = "拼接结果丁戊己\n".repeat(10000);
  const diff = assertRoundTrip(original, draft);
  assert.ok(diff.removed > 0 && diff.added > 0);
});
