# Physics-structured domain shift in quantitative MRI

This repository studies cross-vendor generalization for quantitative MRI (qMRI)
parameter estimation. The authoritative pipeline is the corrected benchmark in
`scripts/run_corrected_benchmark.py`.

## Scientific status

The first draft contained a repeated-seed bug that made many nominally
independent samples identical. Those artifacts are retained only for
provenance. They are not evidence and are not used by the corrected pipeline.

The corrected benchmark provides:

- 100,000 independently seeded MRF signals with sample IDs and physical
  metadata;
- exact signal-duplicate checks before training;
- leave-one-vendor-out evaluation;
- two source vendors during training and a third vendor held out;
- train-only target scaling and source-validation-only hybrid/uncertainty
  calibration;
- ERM, CORAL, GroupDRO, DANN, IRM, and VREx baselines with real source-environment labels;
- paired counterfactual B0/B1/SNR experiments, single-factor holdouts, and
  a compositional unseen-acquisition-combination holdout;
- paired representation analysis and an absolute-error scaling curve;
- optional raw cardiac-MRF phantom validation against independent reference maps.

The benchmark reports absolute held-out-vendor MAE first. DS3 is secondary because
it can become large when source error is small.

## Reproduction

From the repository root:

```bash
bash reproduce.sh
```

The script generates `data/synthetic/mrf_corrected_100k.h5` if necessary,
validates it, runs the corrected benchmark, checks the results, and regenerates
the figures. To run only integrity checks:

bash reproduce.sh verify

To run the held-out acquisition-factor stress test:

bash reproduce.sh factor_holdout

To run the compositional unseen-acquisition-combination holdout:

bash reproduce.sh joint_factor_holdout

To generate paired method effects from an existing benchmark:

bash reproduce.sh method_effects

Dependencies are already installed on the development machine. Set
`MRF_INSTALL_DEPS=1` only when a fresh environment needs installation. The
optional external raw-MRF stage uses an isolated MRpro environment and is run
with `MRPRO_PYTHON=.venv_mrpro/bin/python bash reproduce.sh external`.

Useful controls:

```bash
MRF_EPOCHS=15 MRF_SEEDS=42,123,456 bash reproduce.sh
MRF_ALGOS=erm,coral,groupdro,dann,irm,vrex MRF_SKIP_SCALING=1 bash reproduce.sh
```

## Data and outputs

- Corrected synthetic data: `data/synthetic/mrf_corrected_100k.h5`
- Dataset integrity report: `results/data_validation.json`
- Corrected benchmark: `results/corrected_benchmark.json`
- Independent checks: scripts/verify_paper.py
- External package intake: scripts/validate_external_package.py
- Figures: paper/figures/

The real Zhao--Ma multi-scanner data are used only for map-level
scan-rescan/scanner reproducibility analysis; they are not used to train the
synthetic model and do not provide raw temporal MRF fingerprints or voxelwise
ground truth. Before evaluating any author-supplied package, read
docs/external_validation_protocol.md and run:
bash reproduce.sh validate_external

## Scope and limitations

This is a controlled simulator benchmark, not a clinical validation study.
The scalar Bloch forward model is useful for attribution experiments but is not
a substitute for sequence-accurate Bloch simulation, phantom measurements, or
prospective multi-vendor validation. The paper therefore avoids claims that all
domain-generalization methods fail universally, that synthetic tissue labels
demonstrate clinical harm, or that map-level scanner variation proves model
accuracy.


## Additional public-data audits

The optional analytical validation now includes a four-scanner cMRF release
with independent raw spin-echo reference scans. Run it with:

    MRPRO_PYTHON=.venv_mrpro/bin/python \
    MRPRO_SRC=data/external/mrpro_cmrf/src \
    bash reproduce.sh external_multi

The downloaded VENUS summary is audited separately by
scripts/audit_venus_summary.py. It is contextual qMRI shift evidence only and
is intentionally excluded from neural training and the primary MRF endpoint.
