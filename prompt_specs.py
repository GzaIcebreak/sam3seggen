from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


PartSpec = tuple[str, list[str]]


def normalize_part_specs(entries: str | Sequence[str]) -> list[PartSpec]:
    raw_entries = [entries] if isinstance(entries, str) else list(entries)
    if not raw_entries:
        raise ValueError("at least one component prompt is required")

    specs: list[PartSpec] = []
    for raw in raw_entries:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("component prompt entries must be non-empty strings")
        name, separator, joined = raw.partition("=")
        name = name.strip()
        joined = joined if separator else raw
        raw_concepts = joined.split("+")
        concepts = [concept.strip() for concept in raw_concepts]
        if not name or not concepts or any(not concept for concept in concepts):
            raise ValueError(f"invalid component prompt specification: {raw!r}")
        specs.append((name, concepts))

    duplicates = sorted(
        name for name, count in Counter(name for name, _ in specs).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate component name(s): {duplicates}")
    return specs


def part_names(specs: Sequence[PartSpec]) -> list[str]:
    return [name for name, _ in specs]


def validate_target_name(target: str | None, expected_names: Sequence[str]) -> None:
    if target is not None and target not in expected_names:
        raise ValueError(
            f"unassigned_to {target!r} is not one of the requested components "
            f"{list(expected_names)!r}"
        )


def validate_named_rows(
    expected_names: Sequence[str],
    rows: Sequence[Mapping],
    key: str,
) -> None:
    """Ensure each row names a requested component exactly once."""
    actual: list[str] = []
    for index, row in enumerate(rows):
        if key not in row:
            raise ValueError(
                f"missing field {key!r} in component row at index {index}"
            )
        name = row[key]
        if not isinstance(name, str):
            raise ValueError(
                f"component name in row at index {index} must be a string, got {name!r}"
            )
        actual.append(name)

    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected_names) - set(actual))
    extra = sorted(set(actual) - set(expected_names))
    if duplicates or missing or extra:
        raise ValueError(
            f"component names do not match request: missing={missing}, "
            f"extra={extra}, duplicates={duplicates}"
        )


def legend_part_name(row: Mapping) -> str | None:
    if "part" in row and row["part"] is not None:
        return row["part"]
    return row.get("prompt")


def validate_part_coverage(
    expected_names: Sequence[str],
    rows: Sequence[Mapping],
) -> None:
    """Allow several concept rows to belong to one requested component."""
    actual: list[str] = []
    for index, row in enumerate(rows):
        name = legend_part_name(row)
        if name is None:
            raise ValueError(
                f"missing field 'part' or 'prompt' in component row at index {index}"
            )
        if not isinstance(name, str):
            raise ValueError(
                f"component name in row at index {index} must be a string, got {name!r}"
            )
        actual.append(name)

    missing = sorted(set(expected_names) - set(actual))
    extra = sorted(set(actual) - set(expected_names))
    if missing or extra:
        raise ValueError(
            f"component names do not match request: missing={missing}, extra={extra}"
        )
