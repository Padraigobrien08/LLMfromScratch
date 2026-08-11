import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchAblations } from "./ablationsData";

const respond = (init: { status?: number; type?: string; body?: unknown }) =>
  vi.fn(async () => ({
    status: init.status ?? 200,
    ok: (init.status ?? 200) < 400,
    statusText: "",
    headers: { get: () => init.type ?? null },
    json: async () => init.body,
  })) as unknown as typeof fetch;

afterEach(() => vi.unstubAllGlobals());

describe("fetchAblations", () => {
  it("reads the payload when the file is published", async () => {
    vi.stubGlobal("fetch", respond({ type: "application/json", body: { arms: [] } }));
    expect(await fetchAblations()).toEqual({ arms: [] });
  });

  it("reports absence on a 404", async () => {
    vi.stubGlobal("fetch", respond({ status: 404 }));
    expect(await fetchAblations()).toBeNull();
  });

  // The case a status check alone would get wrong, and the reason this is not just
  // `if (!r.ok)`: a dev server or an SPA fallback answers a missing file with the
  // index page and a cheerful 200.
  it("reports absence when an SPA fallback answers with index.html", async () => {
    vi.stubGlobal("fetch", respond({ status: 200, type: "text/html; charset=utf-8" }));
    expect(await fetchAblations()).toBeNull();
  });

  it("throws on a real server error rather than claiming absence", async () => {
    vi.stubGlobal("fetch", respond({ status: 500, type: "text/plain" }));
    await expect(fetchAblations()).rejects.toThrow();
  });
});
