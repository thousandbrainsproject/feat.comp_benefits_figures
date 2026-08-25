#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONTY_DIR="${SCRIPT_DIR}"
CLI_DIR="${SCRIPT_DIR}/../tbp.cli"

TRAIN_3D=supervised_pre_training_objects_with_stickers_3d_children_3lm_mujoco
TRAIN_2D=supervised_pre_training_objects_with_stickers_2d_children_3lm_mujoco
TRAIN_COMP=supervised_pre_training_objects_with_stickers_comp_models_3lm_mujoco
TRAIN_MONO=supervised_pre_training_objects_with_stickers_monolithic_models_3lm_mujoco
INFER_COMP=base_infer_objects_with_stickers_comp_models_3lm_mujoco
INFER_MONO=base_infer_objects_with_stickers_monolithic_models_3lm_mujoco

OVERWRITE=false
SELF_TEST=false
SESSION_NAME=comp_benefits_pretraining
PROFILE=tbp-experiments
REGION=us-east-2
INSTANCE_NAME="monty-workstation-$(whoami)"
STATUS_DIR="/mnt/results/${USER}/monty/pretrained_models/.run_comp_benefits_pipeline"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Sequentially pretrain 3D children, 2D children, compositional parents, and
monolithic parents on a MuJoCo workstation. After all four checkpoints exist,
launch compositional and monolithic MuJoCo inference on separate stations.

Options:
  --3d-children SELECTOR   Hydra selector for 3D-child pretraining
  --2d-children SELECTOR   Hydra selector for 2D-child pretraining
  --comp-pretrain SELECTOR Hydra selector for compositional pretraining
  --mono-pretrain SELECTOR Hydra selector for monolithic pretraining
  --comp-infer SELECTOR    Hydra selector for compositional inference
  --mono-infer SELECTOR    Hydra selector for monolithic inference
  --cli-dir DIRECTORY      tbp.cli checkout (default: ../tbp.cli)
  --overwrite              Delete the six exact resolved output directories
  --self-test              Run local checks without AWS access
  -h, --help               Show this help

The selected configs control their own logging.run_name. Existing outputs are
never mixed with a rerun: pass --overwrite to replace the exact six run leaves.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_value() {
    [[ -n "${2:-}" && "${2}" != -* ]] || die "$1 requires a value"
}

valid_selector() {
    local value="${1}"
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$ ]] &&
        [[ "/${value}/" != *"/../"* && "/${value}/" != *"/./"* ]]
}

valid_run_name() {
    [[ "${1}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

station_instance_name() {
    printf 'monty-%s-%s' "$(whoami)" "${1//\//_}"
}

self_test() {
    valid_selector supervised_pre_training_objects_with_stickers_3d_children_3lm_mujoco
    valid_selector debug/debug_3lm
    ! valid_selector ../debug_3lm
    ! valid_selector 'debug/debug 3lm'
    valid_run_name debug_3lm
    ! valid_run_name debug/debug_3lm
    [[ "$(station_instance_name debug/debug_3lm)" == "monty-$(whoami)-debug_debug_3lm" ]]
    [[ "/root/debug_3lm" == "/root/$(printf '%s' debug_3lm)" ]]
    echo "Self-test passed."
}

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --3d-children)
            require_value "${1}" "${2:-}"
            TRAIN_3D="${2}"
            shift 2
            ;;
        --2d-children)
            require_value "${1}" "${2:-}"
            TRAIN_2D="${2}"
            shift 2
            ;;
        --comp-pretrain)
            require_value "${1}" "${2:-}"
            TRAIN_COMP="${2}"
            shift 2
            ;;
        --mono-pretrain)
            require_value "${1}" "${2:-}"
            TRAIN_MONO="${2}"
            shift 2
            ;;
        --comp-infer)
            require_value "${1}" "${2:-}"
            INFER_COMP="${2}"
            shift 2
            ;;
        --mono-infer)
            require_value "${1}" "${2:-}"
            INFER_MONO="${2}"
            shift 2
            ;;
        --cli-dir)
            require_value "${1}" "${2:-}"
            CLI_DIR="${2}"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --self-test)
            SELF_TEST=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: ${1}"
            ;;
    esac
done

if ${SELF_TEST}; then
    self_test
    exit 0
fi

for selector in \
    "${TRAIN_3D}" "${TRAIN_2D}" "${TRAIN_COMP}" "${TRAIN_MONO}" \
    "${INFER_COMP}" "${INFER_MONO}"; do
    valid_selector "${selector}" || die "unsafe Hydra selector: ${selector}"
done

requested_cli_dir="${CLI_DIR}"
CLI_DIR="$(cd "${requested_cli_dir}" 2>/dev/null && pwd)" ||
    die "tbp.cli directory not found: ${requested_cli_dir}"
[[ -x "${CLI_DIR}/monty_workstation.sh" ]] || die "missing ${CLI_DIR}/monty_workstation.sh"
[[ -x "${CLI_DIR}/monty_experiment.sh" ]] || die "missing ${CLI_DIR}/monty_experiment.sh"
[[ -x "${CLI_DIR}/efs_workstation.sh" ]] || die "missing ${CLI_DIR}/efs_workstation.sh"
[[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is required by monty_experiment.sh"

workstation() {
    (cd "${CLI_DIR}" && ./monty_workstation.sh "$@")
}

station() {
    (cd "${CLI_DIR}" && ./monty_experiment.sh "$@")
}

source "${CLI_DIR}/scripts/find_instance.sh"

# Start or discover the workstation without touching a possibly active checkout.
workstation up --mujoco --ipv4 --no-rsync --monty-dir "${MONTY_DIR}"

INSTANCE=($(find_instance))
[[ -n "${INSTANCE[*]:-}" ]] || die "failed to find ${INSTANCE_NAME}"
INSTANCE_IPV4="${INSTANCE[3]}"
SSH_TARGET="${USER}@${INSTANCE_IPV4}"
SSH_ARGS=(-o StrictHostKeyChecking=no)

remote() {
    ssh "${SSH_ARGS[@]}" "${SSH_TARGET}" "$@"
}

if remote "tmux has-session -t ${SESSION_NAME} 2>/dev/null"; then
    echo "Found active ${SESSION_NAME}; resuming the wait without rsync or deletion."

    request=""
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        request="$(remote "test -f '${STATUS_DIR}/request.tsv' && cat '${STATUS_DIR}/request.tsv'" 2>/dev/null || true)"
        [[ -n "${request}" ]] && break
        sleep 5
    done
    [[ -n "${request}" ]] || die "active pipeline did not publish ${STATUS_DIR}/request.tsv"

    IFS=$'\t' read -r active_3d active_2d active_comp active_mono active_comp_infer active_mono_infer <<<"${request}"
    [[ "${active_3d}" == "${TRAIN_3D}" &&
       "${active_2d}" == "${TRAIN_2D}" &&
       "${active_comp}" == "${TRAIN_COMP}" &&
       "${active_mono}" == "${TRAIN_MONO}" &&
       "${active_comp_infer}" == "${INFER_COMP}" &&
       "${active_mono_infer}" == "${INFER_MONO}" ]] ||
        die "active pipeline uses different selectors"
else
    for selector in "${INFER_COMP}" "${INFER_MONO}"; do
        saved_instance_name="${INSTANCE_NAME}"
        INSTANCE_NAME="$(station_instance_name "${selector}")"
        existing_station="$(find_instance)"
        INSTANCE_NAME="${saved_instance_name}"
        [[ -z "${existing_station}" ]] ||
            die "inference station already exists for ${selector}; no outputs were deleted"
    done

    # No active job: now it is safe to update the remote checkout.
    workstation up --mujoco --ipv4 --monty-dir "${MONTY_DIR}"

    LOCAL_REMOTE_SCRIPT="$(mktemp)"
    trap 'rm -f "${LOCAL_REMOTE_SCRIPT:-}"' EXIT
    cat >"${LOCAL_REMOTE_SCRIPT}" <<'REMOTE_EOF'
#!/usr/bin/env bash

set -Eeuo pipefail

OVERWRITE="${1}"
TRAIN_3D="${2}"
TRAIN_2D="${3}"
TRAIN_COMP="${4}"
TRAIN_MONO="${5}"
INFER_COMP="${6}"
INFER_MONO="${7}"

PRETRAIN_ROOT="/mnt/results/${USER}/monty/pretrained_models/my_trained_models"
INFER_ROOT="/mnt/results/${USER}/comp_benefits_figures"
STATUS_DIR="/mnt/results/${USER}/monty/pretrained_models/.run_comp_benefits_pipeline"
PLAN_FILE="${STATUS_DIR}/plan.tsv"

mkdir -p "${PRETRAIN_ROOT}" "${INFER_ROOT}" "${STATUS_DIR}"
rm -f "${STATUS_DIR}/running" "${STATUS_DIR}/succeeded" "${STATUS_DIR}/failed" "${PLAN_FILE}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${TRAIN_3D}" "${TRAIN_2D}" "${TRAIN_COMP}" "${TRAIN_MONO}" \
    "${INFER_COMP}" "${INFER_MONO}" >"${STATUS_DIR}/request.tsv"
date -u +%FT%TZ >"${STATUS_DIR}/running"

finish() {
    local code=$?
    rm -f "${STATUS_DIR}/running"
    if [[ ${code} -eq 0 ]]; then
        date -u +%FT%TZ >"${STATUS_DIR}/succeeded"
    else
        printf 'exit_code=%s timestamp=%s\n' "${code}" "$(date -u +%FT%TZ)" >"${STATUS_DIR}/failed"
    fi
    trap - EXIT
    exit "${code}"
}
trap finish EXIT

cd ~/tbp/tbp.monty
source .venv/bin/activate
export MONTY_DATA="/mnt/results/TBP/data"
export MONTY_MODELS="/mnt/results/${USER}/monty/pretrained_models"

python - \
    "${TRAIN_3D}" "${TRAIN_2D}" "${TRAIN_COMP}" "${TRAIN_MONO}" \
    "${INFER_COMP}" "${INFER_MONO}" <<'PY' >"${PLAN_FILE}"
import re
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tbp.monty.hydra import register_resolvers

selectors = sys.argv[1:]
run_names = []
register_resolvers()
config_dir = str(Path("src/tbp/monty/conf").resolve())

with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    for index, selector in enumerate(selectors):
        config = hydra.compose(
            config_name="experiment", overrides=[f"experiment={selector}"]
        )
        target = str(config.experiment._target_)
        OmegaConf.to_container(config, resolve=True)
        run_name = str(config.experiment.config.logging.run_name)

        if index < 4 and not target.endswith(
            ".MontySupervisedObjectPretrainingExperiment"
        ):
            raise SystemExit(f"{selector} is not a supervised-pretraining config")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
            raise SystemExit(f"unsafe logging.run_name for {selector}: {run_name}")

        print(f"{selector}: {run_name}", file=sys.stderr)
        run_names.append(run_name)

if len(set(run_names)) != len(run_names):
    raise SystemExit("all six logging.run_name values must be distinct")

print("\t".join(run_names))
PY

IFS=$'\t' read -r RUN_3D RUN_2D RUN_COMP RUN_MONO RUN_INFER_COMP RUN_INFER_MONO <"${PLAN_FILE}"

for run_name in \
    "${RUN_3D}" "${RUN_2D}" "${RUN_COMP}" "${RUN_MONO}" \
    "${RUN_INFER_COMP}" "${RUN_INFER_MONO}"; do
    [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
        echo "Unsafe resolved run name: ${run_name}" >&2
        exit 1
    }
done

OUTPUT_DIRS=(
    "${PRETRAIN_ROOT}/${RUN_3D}"
    "${PRETRAIN_ROOT}/${RUN_2D}"
    "${PRETRAIN_ROOT}/${RUN_COMP}"
    "${PRETRAIN_ROOT}/${RUN_MONO}"
    "${INFER_ROOT}/${RUN_INFER_COMP}"
    "${INFER_ROOT}/${RUN_INFER_MONO}"
)

existing=()
for output_dir in "${OUTPUT_DIRS[@]}"; do
    [[ -e "${output_dir}" ]] && existing+=("${output_dir}")
done

if [[ ${#existing[@]} -gt 0 && "${OVERWRITE}" != true ]]; then
    echo "Existing outputs require --overwrite:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    exit 1
fi

if [[ "${OVERWRITE}" == true ]]; then
    for output_dir in "${OUTPUT_DIRS[@]}"; do
        case "${output_dir}" in
            "${PRETRAIN_ROOT}/"*|"${INFER_ROOT}/"*) rm -rf -- "${output_dir}" ;;
            *) echo "Refusing unsafe output path: ${output_dir}" >&2; exit 1 ;;
        esac
    done
fi

run_pretraining() {
    local selector="${1}"
    shift
    echo
    echo "=== Running ${selector} ==="
    xvfb-run -a -e /dev/stderr ./run_parallel.py \
        num_parallel=16 \
        "experiment=${selector}" \
        "experiment.config.logging.output_dir=${PRETRAIN_ROOT}" \
        "$@"
}

require_model() {
    [[ -s "${1}/model.pt" ]] || {
        echo "Missing checkpoint: ${1}/model.pt" >&2
        exit 1
    }
}

MODEL_3D="${PRETRAIN_ROOT}/${RUN_3D}/pretrained"
MODEL_2D="${PRETRAIN_ROOT}/${RUN_2D}/pretrained"
MODEL_COMP="${PRETRAIN_ROOT}/${RUN_COMP}/pretrained"
MODEL_MONO="${PRETRAIN_ROOT}/${RUN_MONO}/pretrained"

run_pretraining "${TRAIN_3D}" "experiment.config.model_name_or_path=''"
require_model "${MODEL_3D}"

run_pretraining "${TRAIN_2D}" "experiment.config.model_name_or_path=${MODEL_3D}/"
require_model "${MODEL_2D}"

run_pretraining "${TRAIN_COMP}" "experiment.config.model_name_or_path=${MODEL_2D}/"
require_model "${MODEL_COMP}"

run_pretraining "${TRAIN_MONO}" "experiment.config.model_name_or_path=${MODEL_2D}/"
require_model "${MODEL_MONO}"

echo
echo "Pretraining complete."
printf '  %s\n' "${MODEL_3D}" "${MODEL_2D}" "${MODEL_COMP}" "${MODEL_MONO}"
REMOTE_EOF

    chmod +x "${LOCAL_REMOTE_SCRIPT}"
    REMOTE_SCRIPT="/tmp/run_comp_benefits_pipeline.sh"
    scp -q "${LOCAL_REMOTE_SCRIPT}" "${SSH_TARGET}:${REMOTE_SCRIPT}"

    printf -v REMOTE_COMMAND 'bash %q %q %q %q %q %q %q %q' \
        "${REMOTE_SCRIPT}" "${OVERWRITE}" "${TRAIN_3D}" "${TRAIN_2D}" \
        "${TRAIN_COMP}" "${TRAIN_MONO}" "${INFER_COMP}" "${INFER_MONO}"
    remote "rm -f '${STATUS_DIR}/running' '${STATUS_DIR}/succeeded' '${STATUS_DIR}/failed' && tmux new-session -d -s ${SESSION_NAME} ${REMOTE_COMMAND}"

    echo "Started ${SESSION_NAME} on ${INSTANCE_NAME}."
    echo "Reconnect: cd ${CLI_DIR} && ./monty_workstation.sh connect --mujoco --ipv4 --no-rsync --monty-dir ${MONTY_DIR}"
    echo "Attach:    tmux attach -t ${SESSION_NAME}"
fi

echo "Waiting for sequential pretraining to finish..."
while true; do
    remote_status="$(remote "if test -f '${STATUS_DIR}/succeeded'; then echo succeeded; elif test -f '${STATUS_DIR}/failed'; then echo failed; else echo running; fi" 2>/dev/null || echo unreachable)"
    case "${remote_status}" in
        succeeded)
            break
            ;;
        failed)
            remote "cat '${STATUS_DIR}/failed'" >&2 || true
            die "pretraining failed; workstation left running for inspection"
            ;;
        running)
            remote "tmux has-session -t ${SESSION_NAME} 2>/dev/null" ||
                die "pretraining stopped without a success marker; workstation left running"
            sleep 30
            ;;
        *)
            sleep 30
            ;;
    esac
done

plan="$(remote "cat '${STATUS_DIR}/plan.tsv'")"
IFS=$'\t' read -r RUN_3D RUN_2D RUN_COMP RUN_MONO RUN_INFER_COMP RUN_INFER_MONO <<<"${plan}"
for run_name in \
    "${RUN_3D}" "${RUN_2D}" "${RUN_COMP}" "${RUN_MONO}" \
    "${RUN_INFER_COMP}" "${RUN_INFER_MONO}"; do
    valid_run_name "${run_name}" || die "unsafe remote run name: ${run_name}"
done

PRETRAIN_ROOT="/mnt/results/${USER}/monty/pretrained_models/my_trained_models"
INFER_ROOT="/mnt/results/${USER}/comp_benefits_figures"
MODEL_COMP="${PRETRAIN_ROOT}/${RUN_COMP}/pretrained"
MODEL_MONO="${PRETRAIN_ROOT}/${RUN_MONO}/pretrained"

launch_station() {
    local selector="${1}"
    local run_name="${2}"
    local model_dir="${3}"
    if ! station run_parallel --mujoco -e "${selector}" --monty-dir "${MONTY_DIR}" -- \
        "experiment=${selector}" \
        "experiment.config.logging.output_dir=${INFER_ROOT}" \
        "experiment.config.model_name_or_path=${model_dir}/"; then
        return 1
    fi
    echo "Submitted ${selector}; output: ${INFER_ROOT}/${run_name}"
}

if ! launch_station "${INFER_COMP}" "${RUN_INFER_COMP}" "${MODEL_COMP}"; then
    die "compositional inference was not submitted; workstation left running"
fi

if ! launch_station "${INFER_MONO}" "${RUN_INFER_MONO}" "${MODEL_MONO}"; then
    echo "Retry after resolving the station conflict:" >&2
    printf '  cd %q && ./monty_experiment.sh run_parallel --mujoco -e %q --monty-dir %q -- experiment=%q experiment.config.logging.output_dir=%q experiment.config.model_name_or_path=%q\n' \
        "${CLI_DIR}" "${INFER_MONO}" "${MONTY_DIR}" "${INFER_MONO}" \
        "${INFER_ROOT}" "${MODEL_MONO}/" >&2
    die "monolithic inference was not submitted; compositional inference and workstation left running"
fi

workstation down --monty-dir "${MONTY_DIR}"

cat <<EOF

Pretrained models:
  ${PRETRAIN_ROOT}/${RUN_3D}
  ${PRETRAIN_ROOT}/${RUN_2D}
  ${PRETRAIN_ROOT}/${RUN_COMP}
  ${PRETRAIN_ROOT}/${RUN_MONO}

Inference destinations:
  ${INFER_ROOT}/${RUN_INFER_COMP}
  ${INFER_ROOT}/${RUN_INFER_MONO}

Reconnect to inference stations:
  cd ${CLI_DIR} && ./monty_experiment.sh connect --ipv4 -e ${INFER_COMP} --monty-dir ${MONTY_DIR}
  cd ${CLI_DIR} && ./monty_experiment.sh connect --ipv4 -e ${INFER_MONO} --monty-dir ${MONTY_DIR}

Sync from EFS:
  cd ${CLI_DIR} && ./efs_workstation.sh up
  EFS_HOST="${USER}@REPLACE_WITH_IPV4_FROM_ABOVE"
  rsync -avz "\${EFS_HOST}:${PRETRAIN_ROOT}/${RUN_3D}/" ./pretrained_models/${RUN_3D}/
  rsync -avz "\${EFS_HOST}:${PRETRAIN_ROOT}/${RUN_2D}/" ./pretrained_models/${RUN_2D}/
  rsync -avz "\${EFS_HOST}:${PRETRAIN_ROOT}/${RUN_COMP}/" ./pretrained_models/${RUN_COMP}/
  rsync -avz "\${EFS_HOST}:${PRETRAIN_ROOT}/${RUN_MONO}/" ./pretrained_models/${RUN_MONO}/
  rsync -avz "\${EFS_HOST}:${INFER_ROOT}/${RUN_INFER_COMP}/" ./comp_benefits_figures/${RUN_INFER_COMP}/
  rsync -avz "\${EFS_HOST}:${INFER_ROOT}/${RUN_INFER_MONO}/" ./comp_benefits_figures/${RUN_INFER_MONO}/
  cd ${CLI_DIR} && ./efs_workstation.sh down
EOF
