import { diffChars, diffLines } from "diff";

export type DiffPart = {
  value: string;
  kind: "equal" | "removed" | "added";
  move?: number;
};

export function buildTextDiff(original: string, draft: string) {
  let changes = diffChars(original, draft, { timeout: 60 });
  const coarse = !changes;
  changes ??= diffLines(original, draft, { timeout: 60 });
  const parts: DiffPart[] = changes
    ? changes.map((part) => ({ value: part.value, kind: part.added ? "added" : part.removed ? "removed" : "equal" }))
    : [{ value: original, kind: "removed" }, { value: draft, kind: "added" }].filter((part) => part.value) as DiffPart[];

  // Pair exact moved fragments; other moves still appear as remove + insert.
  const removed = new Map<string, DiffPart[]>();
  for (const part of parts) {
    if (part.kind !== "removed" || Array.from(part.value.trim()).length < 4) continue;
    const candidates = removed.get(part.value) ?? [];
    candidates.push(part);
    removed.set(part.value, candidates);
  }
  let move = 0;
  for (const part of parts) {
    if (part.kind !== "added") continue;
    const source = removed.get(part.value)?.shift();
    if (source) { source.move = ++move; part.move = move; }
  }
  return {
    parts, coarse,
    removed: parts.filter((part) => part.kind === "removed").reduce((sum, part) => sum + Array.from(part.value).length, 0),
    added: parts.filter((part) => part.kind === "added").reduce((sum, part) => sum + Array.from(part.value).length, 0),
  };
}

export type TextDiff = ReturnType<typeof buildTextDiff>;
