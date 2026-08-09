# Data and reproduction guide

The authoritative data product is the corrected synthetic MRF file. It is
created locally and is not downloaded from a third party.

## Corrected synthetic MRF benchmark

From the project root:

```bash
bash reproduce.sh
```

This generates `data/synthetic/mrf_corrected_100k.h5` (approximately 0.76 GB),
validates unique IDs and signal-row hashes, runs the corrected benchmark, and
regenerates the manuscript figures. To run only the integrity checks after the
outputs exist:

```bash
bash reproduce.sh verify
```

The generator uses deterministic SHA-256-derived seeds from the project seed,
sample ID, and domain. The file contains 100,000 signals, 3 vendors, 2 field
strengths, 3 flip-angle schedules, and 2 repetition-time schedules. The legacy
files `mrf_100k.h5`, `mrf_20k.h5`, and `mrf_50k.h5` are retained for provenance
and must not be used for reported results.

## Real multi-scanner maps

The real-data context is the Zhao--Ma whole-brain 3D MRF multi-scanner dataset:
[Zenodo record 8234101](https://zenodo.org/records/8234101).

```bash
mkdir -p data/real
cd data/real
wget https://zenodo.org/records/8234101/files/analysis_images_share.zip
unzip analysis_images_share.zip
cd ../..
```

The project uses those reconstructed maps only for scanner and scan-rescan
reproducibility context. They do not provide raw temporal fingerprints and
voxelwise reference parameters sufficient for validating the synthetic model's
accuracy. Do not use them as a substitute for phantom or raw-MRF validation.

## Optional future validation

BigGABA/MRS loader code is retained for future spectroscopy work and is not
part of the corrected MRF paper. Any additional dataset should be registered
before it is used for model selection or claims of external validity.

## Acquisition-factor holdout

The corrected benchmark also evaluates ERM on field-strength, flip-angle
schedule, and repetition-time schedule levels held out during training. The
source-only split, target scaling, seeds, and primary endpoint match the main
benchmark. Run it with:

    bash reproduce.sh factor_holdout

The output is results/corrected_factor_holdout.json; it is a synthetic
stress test of acquisition shift, not clinical validation.

## Independent raw-MRF phantom validation

The external stage uses the public Open-Source Cardiac MRF phantom record
([Zenodo 15726937](https://zenodo.org/records/15726937)) and the official
[cMRF](https://github.com/PTB-MR/cMRF)/[MRpro](https://github.com/PTB-MR/mrpro)
workflow. It compares dictionary-matched maps with independent nine-tube
reference maps. The files are downloaded into `data/external/` and are not used
for training or tuning the synthetic neural benchmark.

After creating the isolated environment described in
`requirements-external.txt`, run:

```bash
MRPRO_PYTHON=.venv_mrpro/bin/python bash reproduce.sh external
```

The output is `results/external_cmrf_validation.json`, including provenance,
MD5 hashes, reconstruction settings, tube-level MAE/RMSE/bias, and
Bland--Altman limits.


## Four-scanner raw cMRF analytical validation

The public cMRF scanner-comparison release contains raw cMRF acquisitions and
independent raw spin-echo reference scans for nine phantom tubes on four
Siemens scanner configurations. The project stores the downloaded package
under data/external/cmrf_comparison/ and uses the published cMRF/MRpro
reconstruction and EPG dictionary workflow.

Run:

    MRPRO_PYTHON=.venv_mrpro/bin/python \
    MRPRO_SRC=data/external/mrpro_cmrf/src \
    bash reproduce.sh external_multi

This creates results/external_cmrf_multiscanner.json and generated manuscript
macros/tables. The experiment validates analytical reconstruction accuracy
across scanner/sequence implementations; because all four systems are
Siemens, it must not be described as a multi-vendor neural validation.

## VENUS contextual audit

The public VENUS summary is stored under data/external/venus/. The descriptive
audit is run with:

    python3 scripts/audit_venus_summary.py

It creates results/venus_summary_audit.json. The summary contains derived T1,
MTsat, and MTR measurements from paired session labels; it does not contain
raw temporal MRF or independent ground-truth maps in this project. It is not
used for training, model selection, or the primary endpoint.
