import { describe, expect, it } from "vitest";

import { CHAPTER_COUNT, type Route } from "../router";
import { analyticsEnabled, pageviewFor } from "./analytics";

describe("pageviewFor", () => {
  it("reports the front page as the root", () => {
    expect(pageviewFor({ kind: "front" })).toEqual({ route: "/", path: "/" });
  });

  it("groups every chapter under one pattern and keeps the number in the path", () => {
    const views = Array.from({ length: CHAPTER_COUNT }, (_, i) =>
      pageviewFor({ kind: "chapter", n: i + 1 }),
    );

    expect(new Set(views.map((v) => v.route))).toEqual(new Set(["/chapter/[n]"]));
    expect(views.map((v) => v.path)).toEqual([
      "/chapter/1",
      "/chapter/2",
      "/chapter/3",
      "/chapter/4",
      "/chapter/5",
      "/chapter/6",
      "/chapter/7",
      "/chapter/8",
    ]);
  });

  it("names every other page after its route", () => {
    const kinds = [
      "rope",
      "architecture",
      "tests",
      "reproduction",
      "ablations",
      "efficiency",
      "scaling",
      "about",
    ] as const;

    for (const kind of kinds) {
      expect(pageviewFor({ kind } as Route)).toEqual({ route: `/${kind}`, path: `/${kind}` });
    }
  });

  it("gives every page a distinct path, so no two collapse into one row", () => {
    const routes: Route[] = [
      { kind: "front" },
      { kind: "rope" },
      { kind: "architecture" },
      { kind: "tests" },
      { kind: "reproduction" },
      { kind: "ablations" },
      { kind: "efficiency" },
      { kind: "scaling" },
      { kind: "about" },
      ...Array.from({ length: CHAPTER_COUNT }, (_, i): Route => ({ kind: "chapter", n: i + 1 })),
    ];

    const paths = routes.map((route) => pageviewFor(route).path);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("reports a truthy route and path, which is what the component tracks on", () => {
    // `<Analytics>` reports a pageview only when both are truthy, so an empty string for
    // the front page would silently drop the most-visited page on the site.
    for (const route of [{ kind: "front" }, { kind: "chapter", n: 1 }] as Route[]) {
      const view = pageviewFor(route);
      expect(view.route).toBeTruthy();
      expect(view.path).toBeTruthy();
    }
  });
});

describe("analyticsEnabled", () => {
  it("opts out on GitHub Pages, which does not serve the script", () => {
    expect(analyticsEnabled("padraigobrien08.github.io")).toBe(false);
  });

  it("opts in on the custom domain, previews and localhost", () => {
    for (const host of [
      "www.nanogpt-pob.dev",
      "nanogpt-pob.dev",
      "nanogpt-from-scratch.vercel.app",
      "localhost",
    ]) {
      expect(analyticsEnabled(host)).toBe(true);
    }
  });

  it("does not opt out of a domain that merely contains the Pages suffix", () => {
    expect(analyticsEnabled("github.io.nanogpt-pob.dev")).toBe(true);
  });
});
