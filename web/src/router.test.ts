import { describe, expect, it } from "vitest";

import { CHAPTER_COUNT, href, parseRoute, type Route } from "./router";

describe("parseRoute", () => {
  it("reads the front page from an empty or bare hash", () => {
    for (const hash of ["", "#", "#/", "#/ "] as const) {
      expect(parseRoute(hash.trim())).toEqual({ kind: "front" });
    }
  });

  it("reads the three deep-end pages", () => {
    expect(parseRoute("#/rope")).toEqual({ kind: "rope" });
    expect(parseRoute("#/architecture")).toEqual({ kind: "architecture" });
    expect(parseRoute("#/tests")).toEqual({ kind: "tests" });
  });

  it("reads the four results plates", () => {
    expect(parseRoute("#/reproduction")).toEqual({ kind: "reproduction" });
    expect(parseRoute("#/ablations")).toEqual({ kind: "ablations" });
    expect(parseRoute("#/efficiency")).toEqual({ kind: "efficiency" });
    expect(parseRoute("#/scaling")).toEqual({ kind: "scaling" });
  });

  it("reads the about page", () => {
    expect(parseRoute("#/about")).toEqual({ kind: "about" });
    expect(parseRoute("#/about/")).toEqual({ kind: "about" });
  });

  it("reads every chapter", () => {
    for (let n = 1; n <= CHAPTER_COUNT; n++) {
      expect(parseRoute(`#/chapter/${n}`)).toEqual({ kind: "chapter", n });
    }
  });

  it("tolerates a trailing slash", () => {
    expect(parseRoute("#/rope/")).toEqual({ kind: "rope" });
    expect(parseRoute("#/chapter/3/")).toEqual({ kind: "chapter", n: 3 });
  });

  // A chapter number nobody wrote a chapter for is a wrong URL, not a chapter to
  // clamp to: rendering chapter 8 for `#/chapter/99` would quietly disagree with
  // the address bar and with the rail's active state.
  it("falls back to the front page rather than clamping an out-of-range chapter", () => {
    for (const hash of ["#/chapter/0", "#/chapter/9", "#/chapter/-1", "#/chapter/1.5"]) {
      expect(parseRoute(hash)).toEqual({ kind: "front" });
    }
  });

  it("falls back to the front page for anything unrecognised", () => {
    for (const hash of ["#/explainer", "#/chapter", "#/chapter/", "#/rope/extra", "#/nope"]) {
      expect(parseRoute(hash)).toEqual({ kind: "front" });
    }
  });
});

describe("href", () => {
  it("round-trips every route through its own href", () => {
    const routes: Route[] = [
      { kind: "front" },
      { kind: "rope" },
      { kind: "architecture" },
      { kind: "tests" },
      { kind: "reproduction" },
      { kind: "ablations" },
      { kind: "efficiency" },
      { kind: "scaling" },
      ...Array.from({ length: CHAPTER_COUNT }, (_, i) => ({ kind: "chapter", n: i + 1 }) as const),
    ];
    for (const route of routes) {
      expect(parseRoute(href(route))).toEqual(route);
    }
  });
});
