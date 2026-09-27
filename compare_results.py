"""Compare single-turn LLaVA and pruning answers by ID, not accuracy alone."""

import argparse
import json
from pathlib import Path


def read_answers(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        records = ([json.loads(line) for line in stream if line.strip()]
                   if path.suffix == ".jsonl" else json.load(stream))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a nonempty JSON list or JSONL file: {path}")
    answers = {}
    for row in records:
        identifier = row.get("question_id", row.get("id"))
        if identifier is None:
            raise ValueError(f"Missing question_id/id in {path}")
        identifier = str(identifier)
        if not identifier or identifier in answers:
            raise ValueError(f"Empty or duplicate ID {identifier!r} in {path}")
        answer = row.get("answer")
        if answer is None:
            turns = [turn.get("value") for turn in row.get("conversations", [])
                     if turn.get("from") == "gpt"]
            if len(turns) != 1:
                raise ValueError(f"Expected exactly one gpt answer for {identifier} in {path}")
            answer = turns[0]
        if not isinstance(answer, str):
            raise ValueError(f"Non-string answer for {identifier} in {path}")
        answers[identifier] = answer.strip()
    return answers


def compare(baseline, candidate):
    baseline = read_answers(baseline)
    candidate = read_answers(candidate)
    shared = sorted(baseline.keys() & candidate.keys())
    differences = [
        {"id": identifier, "baseline": baseline[identifier], "candidate": candidate[identifier]}
        for identifier in shared if baseline[identifier] != candidate[identifier]
    ]
    missing = sorted(baseline.keys() - candidate.keys())
    extra = sorted(candidate.keys() - baseline.keys())
    return {
        "identical": not differences and not missing and not extra,
        "comparison": "complete answer strings after stripping surrounding whitespace",
        "baseline_samples": len(baseline), "candidate_samples": len(candidate),
        "matched_ids": len(shared), "same_answers": len(shared) - len(differences),
        "different_answers": len(differences), "missing_ids": missing, "extra_ids": extra,
        "differences": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON report (will not overwrite)")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
