#!/usr/bin/env bash
set -euo pipefail

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

DRY_RUN="${DRY_RUN:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${WORK_ROOT}/archive_before_standard/${RATIO_TAG}_seed${SEED}_${STAMP}"

mkdir -p "${BACKUP_ROOT}"

echo "[RESET]"
echo "  ratio:       ${RATIO_TAG}"
echo "  seed:        ${SEED}"
echo "  dry_run:     ${DRY_RUN}"
echo "  backup_root: ${BACKUP_ROOT}"

archive_path() {
    local label="$1"
    local path="$2"

    if [ -z "${path}" ] || [ ! -e "${path}" ]; then
        return 0
    fi

    local resolved
    resolved="$(readlink -m "${path}")"

    case "${resolved}" in
        "${WORK_ROOT}"/*)
            ;;
        *)
            echo "[SKIP OUTSIDE WORK_ROOT] ${resolved}"
            return 0
            ;;
    esac

    local dest="${BACKUP_ROOT}/${label}"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY RUN] mv ${resolved} ${dest}"
        return 0
    fi

    mkdir -p "$(dirname "${dest}")"

    if [ -e "${dest}" ]; then
        dest="${dest}_$(date +%s%N)"
    fi

    mv "${resolved}" "${dest}"
    echo "[ARCHIVED] ${resolved} -> ${dest}"
}

# Stop old background jobs for this ratio.
if [ -d "${PID_DIR}" ]; then
    shopt -s nullglob

    for pid_file in "${PID_DIR}"/*"${RATIO_TAG}"*.pid; do
        pid="$(cat "${pid_file}" 2>/dev/null || true)"

        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            if [ "${DRY_RUN}" = "1" ]; then
                echo "[DRY RUN] kill ${pid} from ${pid_file}"
            else
                kill "${pid}" || true
                echo "[KILLED] ${pid}"
            fi
        fi

        if [ "${DRY_RUN}" != "1" ]; then
            rm -f "${pid_file}"
        fi
    done
fi

# Archive paths defined by 00_common.sh when available.
VARIABLES=(
    SPLIT_DIR
    TEACHER_CKPT_ROOT
    OOF_PRED_ROOT
    OOF_PRED_DIR
    OOF_OUTPUT_DIR
    BOUNDARY_FEEDBACK_DIR
    FEEDBACK_DIR
    FEEDBACK_VIS_DIR
    BEF_PROMPT_JSON
    PROMPT_JSON
    STAGE1_SAVE_DIR
    STAGE1_CKPT_DIR
    STAGE2_SAVE_DIR
    STAGE2_CKPT_DIR
    GEN_OUT_DIR
    GEN_SCORE_CSV
    GEN_ACCEPTED_JSONL
    WEIGHTED_TRAIN_JSON
    FINAL_SEG_SAVE_DIR
    FINAL_EVAL_DIR
)

for var_name in "${VARIABLES[@]}"; do
    if declare -p "${var_name}" >/dev/null 2>&1; then
        value="${!var_name}"

        if [ -n "${value}" ]; then
            archive_path "${var_name}" "${value}"
        fi
    fi
done

# Known fallback locations.
archive_path \
    "teacher_checkpoints_ISIC2018_${RATIO_TAG}" \
    "${WORK_ROOT}/teacher/checkpoints/ISIC2018_${RATIO_TAG}"

archive_path \
    "teacher_oof_predictions_ISIC2018_${RATIO_TAG}" \
    "${WORK_ROOT}/teacher/oof_predictions/ISIC2018_${RATIO_TAG}"

# Archive remaining top-level result files/directories containing ratio tag.
if [ -d "${WORK_ROOT}/results" ]; then
    shopt -s nullglob

    for result_path in "${WORK_ROOT}/results/"*"${RATIO_TAG}"*; do
        archive_path \
            "result_$(basename "${result_path}")" \
            "${result_path}"
    done
fi

echo ""
echo "[DONE] reset scan completed."

if [ "${DRY_RUN}" = "1" ]; then
    echo "Run again with DRY_RUN=0 to perform the archive."
else
    echo "Old outputs archived under:"
    echo "  ${BACKUP_ROOT}"
fi
