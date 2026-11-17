#!/usr/bin/env bash
# Full baler round trip (train -> compress -> decompress -> plot) on a bundled
# project with its config unchanged, followed by tests/roundtrip/check_roundtrip.py.
#
# Run from the repository root (baler resolves workspaces/... relative to CWD):
#     tests/roundtrip/run_roundtrip.sh [WORKSPACE] [PROJECT]
# Uses whatever `python` is on PATH; under uv: `uv run tests/roundtrip/run_roundtrip.sh`.
#
# Writes to $ROUNDTRIP_OUT (default: roundtrip_results/):
#   timings.json, roundtrip_metrics.json, <mode>.log
# If ROUNDTRIP_SUMMARY_MD is set, a markdown summary is appended to that file
# (CI sets it to $GITHUB_STEP_SUMMARY).
#
# Optional:
#   ROUNDTRIP_EPOCHS=N  train for N epochs instead of the config's value. The
#                       config file is edited for the run and restored on exit.
#                       The thresholds are calibrated at the config's epochs, so
#                       they become informational (see below).
#   ROUNDTRIP_THRESHOLDS_INFORMATIONAL=1
#                       report thresholds.json limits but fail only on
#                       structural checks (used by the comparison mode).
set -euo pipefail

WORKSPACE="${1:-CMS_workspace}"
PROJECT="${2:-CMS_project_v1}"
OUT_DIR="${ROUNDTRIP_OUT:-roundtrip_results}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "workspaces/${WORKSPACE}/${PROJECT}" ]]; then
    echo "error: workspaces/${WORKSPACE}/${PROJECT} not found; run from the repo root" >&2
    exit 2
fi

informational="${ROUNDTRIP_THRESHOLDS_INFORMATIONAL:-}"
if [[ -n "${ROUNDTRIP_EPOCHS:-}" ]]; then
    if ! [[ "${ROUNDTRIP_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ROUNDTRIP_EPOCHS must be a positive integer, got '${ROUNDTRIP_EPOCHS}'" >&2
        exit 2
    fi
    config="workspaces/${WORKSPACE}/${PROJECT}/config/${PROJECT}_config.py"
    if ! grep -qE '^    c\.epochs = [0-9]+$' "${config}"; then
        echo "error: no 'c.epochs = <int>' line in ${config}" >&2
        exit 2
    fi
    cp "${config}" "${config}.roundtrip_orig"
    trap 'mv "${config}.roundtrip_orig" "${config}"' EXIT
    sed -i.bak -E "s/^    c\.epochs = [0-9]+$/    c.epochs = ${ROUNDTRIP_EPOCHS}/" "${config}"
    rm -f "${config}.bak"
    echo "=== epochs overridden: $(grep -E '^    c\.epochs = ' "${config}") ==="
    informational=1
fi

mkdir -p "${OUT_DIR}"
timings="${OUT_DIR}/timings.json"
echo "{" > "${timings}"

total_start=$(date +%s)
sep=""
status=0
for mode in train compress decompress plot; do
    echo "=== baler --mode ${mode} ==="
    start=$(date +%s)
    # Capture the status instead of letting set -e exit here, so timings.json is
    # still closed (valid JSON) when a mode fails.
    "${PYTHON}" -m baler --project "${WORKSPACE}" "${PROJECT}" --mode "${mode}" 2>&1 \
        | tee "${OUT_DIR}/${mode}.log" || status=$?
    end=$(date +%s)
    echo "=== ${mode}: $((end - start)) s ==="
    printf '%s  "%s": %d' "${sep}" "${mode}" "$((end - start))" >> "${timings}"
    sep=$',\n'
    if [[ ${status} -ne 0 ]]; then
        printf ',\n  "failed_mode": "%s"' "${mode}" >> "${timings}"
        break
    fi
done
total_end=$(date +%s)
printf ',\n  "total": %d\n}\n' "$((total_end - total_start))" >> "${timings}"

echo "=== per-step wall time (s) ==="
cat "${timings}"

if [[ ${status} -ne 0 ]]; then
    echo "error: baler --mode ${mode} failed (exit ${status}); see ${OUT_DIR}/${mode}.log" >&2
    exit "${status}"
fi

summary_args=()
if [[ -n "${ROUNDTRIP_SUMMARY_MD:-}" ]]; then
    summary_args=(--summary-md "${ROUNDTRIP_SUMMARY_MD}")
fi
if [[ -n "${informational}" ]]; then
    summary_args+=(--thresholds-informational)
fi

"${PYTHON}" tests/roundtrip/check_roundtrip.py \
    --workspace "${WORKSPACE}" --project "${PROJECT}" \
    --metrics-out "${OUT_DIR}/roundtrip_metrics.json" \
    --timings "${timings}" ${summary_args[@]+"${summary_args[@]}"}
