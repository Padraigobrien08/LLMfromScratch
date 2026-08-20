import { describe, expect, it } from "vitest";

import { CHAPTER_BODIES } from "./chapterBodies";
import { CHAPTERS } from "./chapters";
import { CHAPTER_COUNT } from "../router";

/**
 * One chapter count, encoded three ways: the router's route guard, the index the rail
 * and feet read, and the bodies array `Chapter.tsx` indexes with a non-null assertion.
 * Nothing tied them — raise CHAPTER_COUNT for a ninth chapter and forget the body, and
 * `#/chapter/9` renders a crash rather than the front-page fallback `parseRoute`
 * promises. This is the tie.
 */
describe("the chapter count", () => {
  it("is one decision, not three", () => {
    expect(CHAPTERS.length).toBe(CHAPTER_COUNT);
    expect(CHAPTER_BODIES.length).toBe(CHAPTER_COUNT);
  });
});
