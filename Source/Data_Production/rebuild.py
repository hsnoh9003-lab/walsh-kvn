"""Generate publication CSVs in a separate writable Source_Data working copy.

Group definitions come from the verification contracts. Each producer finishes
and, when requested, passes reference comparison before its files are copied
into the working dataset. --partial defers unresolved dependency groups;
--resume checks existing outputs against a separate reference before reuse.
The deposited dataset and the comparison reference are never output targets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import tempfile

if __package__:
    from .verify import CONTRACTS, compare, produce, load_contracts
else:
    from verify import CONTRACTS, compare, produce, load_contracts


def main() -> None:
    """Run dependency passes and populate the separate working dataset.

    Each group is checked before its outputs are copied. Copying is sequential,
    not a rollback transaction. Reference-based resume requires all group
    outputs to compare successfully; partial mode may defer unresolved groups.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--working-source-data",
        type=Path,
        required=True,
        help="Writable working copy containing the retained fixed inputs",
    )
    parser.add_argument(
        "--reference-source-data",
        type=Path,
        help="Optional complete dataset used only to check the generated results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse previously generated working-copy outputs after checking the reference",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Bootstrap independent CSVs before other producer families; defer unresolved groups",
    )
    parser.add_argument(
        "--groups", nargs="+", help="Limit this invocation to named producer groups"
    )
    args = parser.parse_args()
    release = args.release_root.resolve()
    working = args.working_source_data.resolve()
    reference = (
        args.reference_source_data.resolve() if args.reference_source_data else None
    )
    published_data = (release / "Source_Data").resolve()
    if working == published_data:
        parser.error(
            "Use a separate working copy; the deposited Source_Data is read-only"
        )
    if not working.is_dir():
        parser.error("The working Source_Data directory does not exist")
    if reference is not None and working == reference:
        parser.error("The working copy and comparison reference must differ")

    mapping = defaultdict(list)
    for item in (CONTRACTS or load_contracts(working)):
        mapping[item["group"]].append(item)
    if args.groups:
        unknown = set(args.groups) - set(mapping)
        if unknown:
            parser.error(f"Unknown producer groups: {sorted(unknown)}")
        mapping = {group: mapping[group] for group in args.groups}
    # As-printed literature and author-reported values are immutable, deposited
    # inputs; the private manuscript and its extraction script are not released.
    mapping.pop("printed_tables", None)
    remaining = sorted(mapping)
    status = {}
    if args.resume:
        if reference is None:
            parser.error(
                "--resume requires --reference-source-data to validate existing outputs"
            )
        resumed = []
        for group in remaining:
            try:
                for item in mapping[group]:
                    existing = working / item["source"]
                    mode, delta = compare(reference / item["source"], existing)
                    status[item["source"]] = (mode, delta)
            except (FileNotFoundError, ValueError):
                continue
            resumed.append(group)
            print(
                f"Resumed verified {group}: {len(mapping[group])} published paths",
                flush=True,
            )
        remaining = [group for group in remaining if group not in resumed]
    while remaining:
        completed = []
        errors = {}
        for group in remaining:
            with tempfile.TemporaryDirectory(prefix="kvn_rebuild_") as folder:
                output = Path(folder)
                try:
                    produce(group, release, working, None, output)
                    for item in mapping[group]:
                        generated = output / item["output"]
                        if not generated.is_file():
                            raise FileNotFoundError(generated)
                        if reference:
                            original = reference / item["source"]
                            mode, delta = compare(original, generated)
                            status[item["source"]] = (mode, delta)
                    for item in mapping[group]:
                        generated = output / item["output"]
                        destination = working / item["source"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(generated, destination)
                    completed.append(group)
                    print(
                        f"Generated {group}: {len(mapping[group])} published paths",
                        flush=True,
                    )
                except Exception as exc:
                    errors[group] = f"{type(exc).__name__}: {exc}"
        if not completed:
            print("Unresolved producer dependencies:")
            for group in remaining:
                print(f"  {group}: {errors.get(group, 'unknown failure')}")
            if args.partial:
                print(
                    f"Partial bootstrap complete; {len(remaining)} producer groups deferred",
                    flush=True,
                )
                return
            raise SystemExit(1)
        remaining = [group for group in remaining if group not in completed]
    if reference:
        from collections import Counter

        print(
            "Verified:", dict(Counter(mode for mode, _ in status.values())), flush=True
        )
    print(
        f"Generated {sum(len(v) for v in mapping.values())} publication paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
