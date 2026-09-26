import { buildTextDiff } from "../textDiff";

self.onmessage = (event: MessageEvent<{ id: number; original: string; draft: string }>) => {
  const { id, original, draft } = event.data;
  self.postMessage({ id, result: buildTextDiff(original, draft) });
};
