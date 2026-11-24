"""Read saved measurements and present paper results before extra diagnostics.

Summaries never write into a campaign. Figures are explicit derived artifacts;
all aggregation uses the same analysis functions as the experiment's renderer.
"""

from __future__ import annotations

import shutil
import textwrap

ADDITIONAL_NOTICE = (
    "This is additional data, not relevant to or presented in the VLDB paper."
)


def table(title, columns, rows, note=""):
    return dict(title=title, columns=columns, rows=rows, note=note)


def document(title, protocol, paper, additional=(), *, status="Complete"):
    additional = list(additional)
    return dict(
        title=title,
        status=status,
        protocol=protocol,
        paper=list(paper),
        additional=additional,
        additional_notice=ADDITIONAL_NOTICE if additional else "",
    )


def value(number, digits=2):
    return "—" if number is None else f"{number:,.{digits}f}"


def render(report: dict, width: int | None = None) -> str:
    """Aligned, wrapped tables work in a terminal and redirected plain text."""
    width = max(60, width or shutil.get_terminal_size((100, 24)).columns)
    lines = [report["title"], "=" * len(report["title"]), report["status"], ""]
    for note in report["protocol"]:
        lines.extend(textwrap.wrap(note, width=width))
    for heading, key in (
        ("Paper metrics", "paper"),
        ("Additional measurements", "additional"),
    ):
        lines += ["", heading, "-" * len(heading)]
        if key == "additional" and report[key]:
            lines.extend(textwrap.wrap(ADDITIONAL_NOTICE, width=width))
        if not report[key]:
            lines.append("None.")
        for section in report[key]:
            lines += ["", section["title"]]
            if section.get("note"):
                lines.extend(textwrap.wrap(section["note"], width=width))
            columns = section["columns"]
            rows = [[str(cell) for cell in row] for row in section["rows"]]
            if not columns:
                continue
            widths = [
                max([len(str(c)), *(len(r[i]) for r in rows)])
                for i, c in enumerate(columns)
            ]
            while sum(widths) + 3 * (len(widths) - 1) > width and max(widths) > 12:
                i = widths.index(max(widths))
                widths[i] -= 1

            def line(row):
                cells = [
                    textwrap.wrap(
                        str(c), w, break_long_words=True, break_on_hyphens=False
                    )
                    or [""]
                    for c, w in zip(row, widths)
                ]
                return [
                    " | ".join(
                        (c[n] if n < len(c) else "").ljust(w)
                        for c, w in zip(cells, widths)
                    ).rstrip()
                    for n in range(max(map(len, cells)))
                ]

            lines.extend(line(columns))
            lines.append("-+-".join("-" * w for w in widths))
            for row in rows:
                lines.extend(line(row))
    return "\n".join(lines) + "\n"
