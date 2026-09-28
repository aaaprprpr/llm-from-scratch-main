import { diffArrays, diffChars, diffLines } from "diff";

export type DiffPart = {
  value: string;
  kind: "equal" | "removed" | "added";
  move?: number;
};

function paragraphs(text: string): string[] {
  return text.match(/[\s\S]*?(?:\r?\n\r?\n|$)/g)?.filter(Boolean) ?? [];
}

export function buildTextDiff(original: string, draft: string) {
  const parts: DiffPart[] = [];
  const append = (kind: DiffPart["kind"], value: string) => {
    if (!value) return;
    const last = parts.at(-1);
    if (last?.kind === kind) last.value += value;
    else parts.push({ kind, value });
  };

  // Anchor unchanged paragraphs first. A global character diff can match repeated
  // words across distant sections and make unrelated sentences appear interleaved.
  const changes = diffArrays(paragraphs(original), paragraphs(draft), { timeout: 60 });
  let coarse = !changes;
  if (changes) {
    let removed: string[] = [];
    let added: string[] = [];
    const flush = () => {
      if (removed.length === added.length && removed.length > 0 && removed.length <= 8) {
        for (let index = 0; index < removed.length; index++) {
          const before = removed[index];
          const after = added[index];
          const local = before.length + after.length <= 6000
            ? diffChars(before, after, { timeout: 40 }) : undefined;
          const common = local?.filter((part) => !part.added && !part.removed)
            .reduce((sum, part) => sum + part.value.length, 0) ?? 0;
          if (local && common >= Math.min(before.length, after.length) * .4) {
            for (const part of local) append(part.added ? "added" : part.removed ? "removed" : "equal", part.value);
          } else {
            append("removed", before);
            append("added", after);
          }
        }
      } else {
        const before = removed.join("");
        const after = added.join("");
        const local = before && after && before.length + after.length <= 6000
          ? diffChars(before, after, { timeout: 40 }) : undefined;
        const common = local?.filter((part) => !part.added && !part.removed)
          .reduce((sum, part) => sum + part.value.length, 0) ?? 0;
        if (local && common >= Math.min(before.length, after.length) * .5) {
          for (const part of local) append(part.added ? "added" : part.removed ? "removed" : "equal", part.value);
        } else {
          append("removed", before);
          append("added", after);
        }
      }
      removed = [];
      added = [];
    };
    for (const change of changes) {
      if (change.removed) removed.push(...change.value);
      else if (change.added) added.push(...change.value);
      else { flush(); append("equal", change.value.join("")); }
    }
    flush();
  } else {
    // Large inputs still show both complete versions if paragraph matching times out.
    const fallback = diffLines(original, draft, { timeout: 60 });
    if (fallback) for (const part of fallback) append(part.added ? "added" : part.removed ? "removed" : "equal", part.value);
    else { append("removed", original); append("added", draft); }
  }

  const removedByText = new Map<string, DiffPart[]>();
  for (const part of parts) {
    if (part.kind !== "removed" || Array.from(part.value.trim()).length < 4) continue;
    const candidates = removedByText.get(part.value) ?? [];
    candidates.push(part);
    removedByText.set(part.value, candidates);
  }
  let move = 0;
  for (const part of parts) {
    if (part.kind !== "added") continue;
    const source = removedByText.get(part.value)?.shift();
    if (source) { source.move = ++move; part.move = move; }
  }
  return {
    parts, coarse,
    removed: parts.filter((part) => part.kind === "removed").reduce((sum, part) => sum + Array.from(part.value).length, 0),
    added: parts.filter((part) => part.kind === "added").reduce((sum, part) => sum + Array.from(part.value).length, 0),
  };
}

export type TextDiff = ReturnType<typeof buildTextDiff>;
