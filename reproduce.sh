#!/usr/bin/env bash
# Reproduce the corrected, leakage-free benchmark.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

if [[ "${1:-}" == "verify" ]]; then
  python3 scripts/verify_paper.py
  exit 0
fi

if [[ "${1:-}" == "factor_holdout" ]]; then
  python3 scripts/run_factor_holdout.py
  exit 0
fi

if [[ "${1:-}" == "joint_factor_holdout" ]]; then
  python3 scripts/run_joint_factor_holdout.py
  exit 0
fi

if [[ "${1:-}" == "recompute_analyses" ]]; then
  python3 scripts/recompute_corrected_analyses.py
  exit 0
fi

if [[ "${1:-}" == "method_effects" ]]; then
  python3 scripts/analyze_benchmark_effects.py
  exit 0
fi

if [[ "${1:-}" == "validate_external" ]]; then
  external_root="${MRF_EXTERNAL_ROOT:-$PWD/data/external/external_mrf_package}"
  strict_args=()
  if [[ "${MRF_EXTERNAL_STRICT:-0}" == "1" ]]; then
    strict_args+=(--strict)
  fi
  python3 scripts/validate_external_package.py \
    --root "$external_root" \
    --output results/external_package_validation.json \
    "${strict_args[@]}"
  exit 0
fi

if [[ "${1:-}" == "external" ]]; then
  external_python="${MRPRO_PYTHON:-$PWD/.venv_mrpro/bin/python}"
  if [[ ! -x "$external_python" ]]; then
    echo "External validation requires MRPRO_PYTHON or .venv_mrpro/bin/python" >&2
    exit 1
  fi
  "$external_python" scripts/run_external_cmrf_validation.py
  exit 0
fi

if [[ "${1:-}" == "external_multi" ]]; then
  external_python="${MRPRO_PYTHON:-$PWD/.venv_mrpro/bin/python}"
  external_src="${MRPRO_SRC:-$PWD/data/external/mrpro_cmrf/src}"
  if [[ ! -x "$external_python" ]]; then
    echo "Multi-scanner validation requires MRPRO_PYTHON or .venv_mrpro/bin/python" >&2
    exit 1
  fi
  if [[ ! -d "$external_src" ]]; then
    echo "Multi-scanner validation requires MRPRO_SRC or data/external/mrpro_cmrf/src" >&2
    exit 1
  fi
  PYTHONPATH="$external_src${PYTHONPATH:+:$PYTHONPATH}" \
    "$external_python" scripts/run_external_cmrf_multiscanner.py
  exit 0
fi

if [[ "${MRF_INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r requirements.txt
fi

if [[ ! -f data/synthetic/mrf_corrected_100k.h5 ]]; then
  python3 scripts/generate_corrected_data.py --output data/synthetic/mrf_corrected_100k.h5
fi

python3 scripts/validate_dataset.py \
  data/synthetic/mrf_corrected_100k.h5 \
  --output results/data_validation.json
python3 scripts/run_full_benchmark.py
python3 scripts/run_factor_holdout.py
python3 scripts/run_joint_factor_holdout.py
python3 scripts/analyze_benchmark_effects.py
python3 scripts/verify_paper.py
python3 scripts/generate_figures.py

echo "Corrected reproduction completed."
