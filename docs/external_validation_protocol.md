# Frozen external temporal-MRF validation protocol

This contract is for the missing experiment: evaluation of the already-frozen
neural model on sequence-compatible temporal MRF data from independent
subjects, scanners, or sites. It is separate from the public map-only and
analytical cMRF audits already in this repository.

## Required package

The author package should contain:

    external_mrf_package/
      manifest.json
      samples.csv
      fingerprints/
      schedules/
      references/
      metadata/

manifest.json must follow
docs/external_mrf_manifest_template.json. samples.csv must follow
docs/external_mrf_samples_template.csv. All paths in the CSV are relative to
the package root.

Each scan row must provide:

- complex temporal fingerprint data, not only final T1/T2 maps;
- the exact FA/TR/TE schedule and the number of time points;
- independent T1 and T2 reference measurements, with their acquisition method;
- subject, scan, repeat, site, vendor, scanner-model, field-strength, and
  sequence identifiers;
- a license or data-use statement and a citable persistent source;
- a frozen-external-test split label.

For the current neural model, the fingerprint length must be exactly 1,000
time points. Resampling, truncation, padding, or schedule substitution must be
treated as a new preprocessing experiment and cannot be performed silently.

## Recommended evidence strength

The preferred package has at least 20 independent subjects, at least two
vendors and sites, scanner-level metadata, repeated scans or a multi-scan
phantom, spatial B0/B1 maps, noise or SNR metadata, and independent reference
maps acquired with a separate sequence. A smaller calibrated phantom is still
valuable, but it should be labeled as a phantom/analytical validation rather
than a subject-level clinical validation.

Final maps without temporal fingerprints can support scan-rescan or scanner
reproducibility context. They cannot validate the neural inverse model. A
single-vendor package cannot support a cross-vendor neural claim, even if it
has many subjects.

## Freeze and leakage controls

Before opening external reference labels:

1. Freeze the checkpoint, architecture, input normalization, seeds, optimizer
   settings, method-selection rule, and primary endpoint.
2. Compute and record SHA-256 hashes for the checkpoint, configuration, code
   revision, and package files.
3. Run the intake validator with the strict option.
4. Create the final subject/scanner/site holdout before reading reference values.
5. Never fit target scaling, thresholds, hybrid weights, uncertainty
   calibration, or hyperparameters on the external target labels.
6. Keep every subject, repeat, and scanner identity in one split only. A
   scan-rescan repeat is not an independent subject.

The external package is evaluation-only. If an exploratory preprocessing choice
is made after seeing target errors, that result must be labeled exploratory and
cannot replace the frozen primary analysis.

## Evaluation endpoints

The primary endpoint should be absolute MAE in milliseconds, reported separately
for T1 and T2. Also report RMSE, signed bias, median absolute error, and
subject- or tube-clustered 95% confidence intervals. For repeats, report
within-subject repeatability, coefficient of variation where appropriate, and
Bland-Altman bias/limits. For scanner/site comparisons, report the number of
independent subjects per cluster rather than treating voxels as independent.

If uncertainty is evaluated, calibration must be frozen from source validation
only. Report empirical coverage, interval width, and error-stratification on
the external set; do not tune a target quantile after observing external
coverage.

## Interpretation boundary

A clean external result supports the stated cohort and sequence. It does not
prove universal cross-vendor or clinical generalization. A poor result is also
scientifically useful if the input schedule, reference method, and failure mode
are traceable. The paper should distinguish:

- analytical reconstruction agreement;
- phantom repeatability;
- external temporal-MRF neural accuracy; and
- prospective clinical utility.

The current public Zhao--Ma release and the SAFE 3D MRF release are useful
sources to inspect, but their published contents are map-centered rather than
a drop-in guarantee of raw 1,000-point temporal fingerprints. The author
package must therefore be validated rather than accepted by filename.

## Commands

With the package placed at the default location:

    python3 scripts/validate_external_package.py
    python3 scripts/validate_external_package.py --strict

To use another location:

    python3 scripts/validate_external_package.py \
      --root /path/to/external_mrf_package \
      --output results/external_package_validation.json \
      --strict

A missing package is reported as status: missing without failing the
non-strict command. An existing package with missing requirements is reported
as status: invalid. No training or model selection is performed by this
validator.
