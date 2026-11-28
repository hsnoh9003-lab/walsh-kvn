# Third-party scientific software

This document identifies upstream scientific packages and external inputs
used by the reproduction paths, with licence checks dated 26 November 2026.
Original Walsh–KvN code and code documentation are licensed under Apache-2.0
as specified in LICENSE and NOTICE. Separately installed dependencies retain
their own licences; this is not an inventory of every Python dependency.

| Upstream software | Published licence | Project use |
| --- | --- | --- |
| [Strilanc/magic-state-cultivation](https://github.com/Strilanc/magic-state-cultivation), accompanying “Magic state cultivation: growing T states as cheap as CNOT gates” | [Apache-2.0](https://github.com/Strilanc/magic-state-cultivation/blob/main/LICENSE) | Circuit generation, sampling and gap-threshold processing through an external checkout and its `cultiv` package. |
| [litinski/magicstates](https://github.com/litinski/magicstates), accompanying “Magic State Distillation: Not as Costly as You Think” | [Apache-2.0](https://github.com/litinski/magicstates/blob/master/LICENSE) | Numerical factory functions imported from a separately supplied upstream `Python/` directory. |

## Cultivation

[`Source/Gidney/generate_raw_stats.py`](Source/Gidney/generate_raw_stats.py)
accepts `--cultiv-repo`, calls that checkout's `tools/make_circuits`, and uses its
custom decoder through Sinter. [`Source/Gidney/build_tradeoff.py`](Source/Gidney/build_tradeoff.py)
imports `cultiv` and calls `cultiv.split_by_gap_threshold` to process the raw
statistics. The upstream `cultiv` implementation is an external prerequisite
for these entry points, not supplied by the `Source/Gidney/` directory.

## Distillation

[`Source/RE/generate_nac_factory_catalogue.py`](Source/RE/generate_nac_factory_catalogue.py)
accepts `--litinski-python` and imports the upstream numerical functions from
that directory. Its documentation describes the breakpoint suppression and
scalar-argument compatibility wrapper used around those external functions.
The upstream source files are not included in this generator's directory.

[`Source/RE/Litinski.py`](Source/RE/Litinski.py) reads the archived numerical
factory results and applies the common resource model. Its “A Game” branch
separately implements formulae from D. Litinski, “A Game of Surface Codes,”
Quantum 3, 128 (2019), as identified in the module. A paper citation and a
software licence serve different purposes; the cited paper is not a blanket
licence declaration for this repository.

## When redistributing upstream code

If upstream source or a modified copy is redistributed, comply with
[Apache License 2.0, section 4](https://www.apache.org/licenses/LICENSE-2.0):
include a copy of the licence, retain applicable copyright and attribution
notices, identify modified files, and retain applicable upstream `NOTICE`
content if supplied. A repository link or paper citation alone does not
replace these requirements. The licence selected for original project code
does not remove these obligations for upstream material.

## QuCF reference inputs: retrieved separately

The KvN–LCHS comparison uses 32 original input files from
[QuCF/QuCF, revision f34f89c24eff4ab79d3ecdea72da3cec88dce94f](https://github.com/QuCF/QuCF/tree/f34f89c24eff4ab79d3ecdea72da3cec88dce94f).
They comprise 12 `.oracle` circuit descriptions and 20 `.hdf5` inputs/results.
No redistribution licence for these files was identified in the checked
repository tree, README or inspected source files. The QuEST submodule's
licence is not treated as a licence for all QuCF files.

These original files are excluded from both the code and Source Data releases.
[`Source/qucf_inputs.py`](Source/qucf_inputs.py) retrieves them from the exact
upstream commit into a local cache, verifies each byte count and SHA-256 digest,
and reuses verified files offline. It does not grant an upstream licence.
The existing [input manifest](../Source_Data/Tables/Table_S11/LCHS_source_counts/upstream_input_manifest.json)
records the 32 inputs and two C++ engine files used for provenance; those engine
implementations are also not bundled. The publication's resource tables and
the project's reconstruction/validation code remain in the release.
