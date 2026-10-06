import argparse
from pathlib import Path

from atr.eval.report import compare_runs, format_rows, load_records
from atr.eval.run import RunInfo


def _info(run_dir: Path) -> RunInfo:
    return RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Paired difference between two runs (B minus A), from their saved records."
    )
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    a, b = _info(args.a), _info(args.b)
    if (a.evalset_hash, a.corpus_pin, a.split) != (b.evalset_hash, b.corpus_pin, b.split):
        raise SystemExit("the two runs differ in question set, corpus or split; not comparable")
    rows = compare_runs(load_records(args.a), load_records(args.b), args.resamples, args.seed)
    print(f"B minus A | A = {a.name} | B = {b.name} | split {a.split}")
    print(
        f"Paired bootstrap over question groups, {args.resamples} resamples, seed {args.seed}. "
        "* marks a 95% interval that excludes zero.\n"
    )
    print(format_rows(rows))


if __name__ == "__main__":
    main()
