"""A pytest plugin that collects the tests chosen for the site's test page.

The `#/tests` page is not "340 tests pass" — a count is a claim about effort, not about
correctness, and a reader has no way to check it. What is worth showing is *what a
handful of the tests assert*, and the specific bug each one exists to catch.

The obvious way to build that page is to type the rows into a TypeScript file, which is
the thing this repository keeps proving it should not do: a renamed or deleted test would
leave the site advertising coverage that no longer exists, and nothing would fail. So the
rows come from the tests themselves, via a marker:

    @pytest.mark.showcase(
        pins="what the test asserts",
        why="the bug it exists to catch",
    )

Curation stays explicit — only marked tests appear, so the page is a chosen argument
rather than a directory listing — while the *existence* of each row is checked by
collection. Delete the test and the row disappears; rename it and the committed export
stops matching, which fails CI.

Run as a plugin during an ordinary collection:

    pytest tests --collect-only -p llmfs.export.showcase --showcase-json=out.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MARKER = "showcase"


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--showcase-json",
        default=None,
        help="write the marked tests to this path as JSON, then continue as normal",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}(pins, why): show this test on the site's test page, with what it pins "
        "and the bug it exists to catch",
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    destination = config.getoption("--showcase-json")
    if not destination:
        return

    rows: dict[str, dict[str, Any]] = {}
    for item in items:
        marker = item.get_closest_marker(MARKER)
        if marker is None:
            continue

        # A parametrized test is many items and one claim. Keyed on the function's
        # nodeid so the page shows the claim once, with the number of cases behind it —
        # which is itself worth printing: "run against all ten architecture variants"
        # is a stronger statement than the same sentence with no count.
        node = item.nodeid.split("[")[0]
        file, _, name = node.partition("::")
        row = rows.setdefault(
            node,
            {
                "file": file,
                "name": name,
                "pins": marker.kwargs.get("pins", ""),
                "why": marker.kwargs.get("why", ""),
                "cases": 0,
            },
        )
        row["cases"] += 1

    payload = sorted(rows.values(), key=lambda r: (r["file"], r["name"]))
    Path(destination).write_text(json.dumps(payload, indent=2) + "\n")
