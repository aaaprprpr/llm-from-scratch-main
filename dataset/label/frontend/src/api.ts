const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";

export async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? `${response.status} ${response.statusText}`;
    throw new Error(String(detail));
  }
  return body as T;
}
