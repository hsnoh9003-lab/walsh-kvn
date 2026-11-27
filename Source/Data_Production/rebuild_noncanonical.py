"""Regenerate additional publication tables using the producer map.

Inputs
------
A writable working dataset with its fixed scientific inputs; optional separate reference dataset.

Outputs
-------
Mapped generated tables in --working-source-data, with group-level progress.

Run from the repository root
----------------------------
python -m Source.Data_Production.rebuild_noncanonical --working-source-data ../Source_Data_Reproduced

Scope
-----
Used by end_to_end in multiple dependency passes. Raw Gidney cultivation records
are retained experimental/calibration inputs. With a comparison reference,
--resume verifies every mapped file before reuse; without one, it checks only
file presence and a nonempty header. Presence-only reuse is not verification.
"""

from __future__ import annotations

# Direct-file execution must locate Codes/ independently of the working directory.
if __package__ in (None, ""):
    import sys as _path_sys
    from pathlib import Path as _Path
    _codes = next(p for p in _Path(__file__).resolve().parents
                  if (p / "Source/data_locations.py").is_file())
    _path_sys.path.insert(0, str(_codes))


import argparse
from collections import Counter
from pathlib import Path
import sys

from Source.Data_Production.source_data_rebuild import (
    CODES,
    all_routes,
    publish,
    run_producer,
    RUN_COMMANDS,
)


def main() -> None:
    """Run ready producer groups, publish completed outputs and optionally defer unresolved dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-source-data", type=Path, required=True)
    parser.add_argument(
        "--reference-source-data",
        type=Path,
        help="Optional immutable dataset for output comparison",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Complete independent groups and defer missing cross-family dependencies",
    )
    args = parser.parse_args()
    working = args.working_source_data.resolve()
    reference = (
        args.reference_source_data.resolve() if args.reference_source_data else None
    )
    if working == (CODES.parent / "Source_Data").resolve():
        parser.error("Refusing to replace the published Source_Data directory")
    if reference and working == reference:
        parser.error("Working and reference directories must differ")
    groups = sorted(RUN_COMMANDS.keys() & {r.group for r in all_routes()})
    remaining = list(groups)
    failures = {}
    # Retry deferred groups only while other groups make progress. A partial
    # pass may leave dependencies unresolved; a complete pass must resolve all.
    while remaining:
        completed = []
        for group in remaining:
            routes = [r for r in all_routes() if r.group == group]
            if args.resume:
                from Source.Data_Production.verify import compare
                from Source.Data_Production.source_data_rebuild import file_header

                try:
                    for route in routes:
                        target = working / route.published_in_source_data
                        if reference:
                            compare(reference / route.published_in_source_data, target)
                        else:
                            if not target.is_file() or not file_header(target):
                                raise FileNotFoundError(target)
                except (ValueError, FileNotFoundError, StopIteration):
                    pass
                else:
                    if reference:
                        print(
                            f"Already verified {group}: {len(routes)} paths", flush=True
                        )
                    else:
                        print(
                            f"Existing outputs reused for {group}: {len(routes)} paths; "
                            "no reference comparison",
                            flush=True,
                        )
                    completed.append(group)
                    continue
            try:
                run_producer(group, working)
                # This is an explicit writable staging dataset, never the
                # deposited source archive. Allow restarting an interrupted
                # end-to-end computation without a stale output blocking it.
                publish(group, working, overwrite=True)
                if reference:
                    from Source.Data_Production.verify import compare

                    for route in routes:
                        compare(
                            reference / route.published_in_source_data,
                            working / route.published_in_source_data,
                        )
                completed.append(group)
                print(f"Regenerated {group}: {len(routes)} paths", flush=True)
            except Exception as exc:
                failures[group] = f"{type(exc).__name__}: {exc}"
                print(f"Deferred {group}: {failures[group]}", flush=True)
        if not completed:
            print("Unresolved producer groups:", flush=True)
            for group in remaining:
                print(f"  {group}: {failures[group]}", flush=True)
            if args.partial:
                print(
                    f"Partial noncanonical stage complete; {len(remaining)} groups deferred",
                    flush=True,
                )
                return
            raise SystemExit(1)
        remaining = [group for group in remaining if group not in completed]
    print(
        f"Regenerated {sum(1 for r in all_routes() if r.group in RUN_COMMANDS)} mapped publication paths"
    )


if __name__ == "__main__":
    main()
