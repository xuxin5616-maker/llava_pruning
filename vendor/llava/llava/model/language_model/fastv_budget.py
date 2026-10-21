"""Spatial candidate groups from the ACTUAL dense AnyRes packing geometry."""


def base_anyres_groups(base_count, rows, cols, *, order="base_first", final_newline=False):
    if min(base_count, rows, cols) < 1 or order not in {"base_first", "anyres_first"}:
        raise ValueError("Invalid Base/AnyRes budget geometry")
    highres_length = rows * (cols + 1)
    moved_tail = order == "anyres_first" and not final_newline
    base_offset = highres_length - int(moved_tail) if order == "anyres_first" else 0
    anyres_offset = 0 if order == "anyres_first" else base_count
    # Row newlines (including the moved tail) and an optional final newline
    # deliberately belong to neither spatial candidate group.
    return {
        "packed_tokens": base_count + highres_length + int(final_newline),
        "groups": [
            {"name": "base", "indices": list(range(base_offset, base_offset + base_count))},
            {"name": "anyres", "indices": [anyres_offset + row * (cols + 1) + col
                                           for row in range(rows) for col in range(cols)]},
        ],
    }
