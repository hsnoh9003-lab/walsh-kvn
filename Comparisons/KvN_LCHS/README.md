# Novikau–Joseph ten-step LCHS–KvN comparison

The manuscript estimates the Fig. 14 circuit at normalized time 1 with 128 position cells and ten steps of length 0.1. Seven Hadamards on the position register prepare equal amplitudes. The tally includes these gates once, together with the separate weight preparation, controlled selection and evolution. The accompanying 20-step tally provides a source check.

## Reproduce the inputs and counts

With NumPy and h5py installed as listed in `source_replay/requirements_source_audit.txt`, run the following from the package root.

    python Comparisons/KvN_LCHS/verify.py

The tally reconstructs small block encodings from the pinned QuCF definitions and angles and counts the nested QSP, selection and repeated blocks algebraically. It checks two encoded matrices against the saved block-encoding outputs. The ten-step total is independently stored in `basic/N_gates` in `Source_Data/Shared_source_data/RE_common/LCHS_source_inputs/dt1/kvn_OUTPUT.hdf5`. The zero-ancilla output amplitudes in that file give a nominal success probability of 0.8762773928256311. The authors report approximately 0.88. This verification does not rerun the full 25-qubit state evolution.

The script compares the regenerated histograms with the archived CSVs, then compiles the regenerated inventories. Each row specifies a gate, its positive and negative control counts, any angles and its multiplicity. For the ten-step circuit, 1,566 distinct rows represent 108,018,555 occurrences. Their order in the original circuit cannot be recovered from the histogram alone.

The manuscript uses `compile_stmc_one_clean.py` with the shared gate decomposition in `Source/KvN/one_clean_synthesis.py`. One clean work qubit is reused throughout, giving 26 logical qubits. Full control-branch phases are retained. The compiler checks the source primitives on superpositions and tests that the work qubit returns to zero. After weighting each decomposition by its source multiplicity, the inventory contains 8,781,048,741 Cliffords (including 5,461,507,502 CNOTs), 5,914,351,326 native T or T† gates and 307,305,994 arbitrary Z rotations. Inter-gate cancellation across source primitives is outside this compilation. The tests and the archived gate totals give no estimate of how much further global optimization might save.

The width, native T and arbitrary-rotation counts, angle-rounding bound and success probability are checked against `Source_Data/Shared_source_data/RE_common/RE/workloads_v38.json`. The resource estimator then selects synthesis precision for each feasible hardware design. At the reference physical error probability, the fastest uncapped design uses 126 T gates per rotation, giving 5,914,351,326 + 126 × 307,305,994 = 44,634,906,570 T gates per attempt. Table S10 also accounts for retries when collecting 815 accepted shots. The October archive corroborates the original circuit tally. Its earlier compiler and resource estimates describe separate configurations.

## Generator and archive check

The independent check reconstructs the 128-cell sine-drift generator and evaluates the effect of a scalar shift. From the package root, run the following with NumPy and SciPy. For the optional comparison against the included QuCF HDF5 arrays, install h5py and add the archive argument as shown on the second line.

    python Comparisons/KvN_LCHS/verify_lchs_sine_shift.py
    python Comparisons/KvN_LCHS/verify_lchs_sine_shift.py --qucf-dir "/path/to/Source_Data/Shared_source_data/RE_common/LCHS_source_inputs"

The optional `--qucf-dir` path may be obtained with
`python -m Source.data_locations --download`; the principal `verify.py`
command already resolves the DOI source automatically.

## Provenance

- QuCF revision f34f89c24eff4ab79d3ecdea72da3cec88dce94f, branch KvN-LCHS. https://github.com/QuCF/QuCF/tree/f34f89c24eff4ab79d3ecdea72da3cec88dce94f
- Notebook revision 2727acd977476ab2151a5b12e6765e0a219f8293. https://github.com/QuCF/scripts-py/tree/2727acd977476ab2151a5b12e6765e0a219f8293
- I. Novikau and I. Joseph, Computer Physics Communications 309, 109498 (2025), Sec. VI.3 and Fig. 14. https://doi.org/10.1016/j.cpc.2025.109498
- The 32 pinned original input files match their hashes in `source_replay/upstream_input_manifest.json`. The manifest also identifies two upstream C++ engine files. The archived tallies and compilation report are in `Source_Data/Tables/Table_S10/LCHS_source_counts/`; `source_verification.json` records the audit boundary.
