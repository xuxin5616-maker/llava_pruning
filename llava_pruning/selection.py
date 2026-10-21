"""Select visualization IDs independently from the evaluation sample set."""


def visualization_key(value):
    return str(value).removeprefix("sample_")


def resolve_visualization_ids(values, samples):
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError("visualize_samples must be a list of sample IDs")
    selected = frozenset(visualization_key(value) for value in values)
    if not selected or "" in selected:
        raise ValueError("visualize_samples must contain nonempty sample IDs")
    available = {visualization_key(sample.sample_id) for sample in samples}
    missing = selected - available
    if missing:
        raise ValueError(f"Visualization question_id/id not found in input: {sorted(missing)}")
    return selected
