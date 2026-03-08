#!/bin/bash
#
# 8-node proxy of full 64-node SFT — HYBRIDEP + bindpcie, **no CUDA graph** (baseline).
# Same as megaron_sft_proxy_hybridep_bindpcie.sh but with CUDA graph options removed:
#   removed: --cuda-graph-scope mamba attn moe_router --te-rng-tracker
# Use this for baseline experiments (e.g. compare with _cudagraph_attn_mamba).
#
#   [+] scripts/bindpcie.sh: CPU/NUMA binding per rank (--cpu=node --mem=node)
#   recompute: selective / moe only
#
# Requires: numactl inside container; bindpcie.sh at MEGATRON_LM_DIR/scripts/bindpcie.sh
# Logs/ckpts: sft-runs/proxy-sft-hybridep-bindpcie-no-cg/

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --time=3:45:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --segment=8
#SBATCH --job-name=proxy-sft-hybridep-bindpcie-no-cg

################################################################
### TransformerEngine
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_CPU_OFFLOAD_V1=1
export TORCHINDUCTOR_WORKER_START=fork

################################################################
### HybridEP / MNNVL (required for --moe-flex-dispatcher-backend hybridep)
################################################################
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16
export USE_MNNVL=1

################################################################
### UCX (prevents memory hook conflicts in multi-node)
################################################################
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=n
export UCX_ERROR_SIGNALS=

################################################################
### General
################################################################
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-LM/sft-runs"
MEGATRON_LM_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-LM"
IMAGE="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/containers/pt_ultra_mamba_ssmv230_23jan28.sqsh"

# bindpcie: CPU/mem binding script (expects LOCAL_RANK or SLURM_LOCALID per task)
BINDPCIE_SCRIPT="${MEGATRON_LM_DIR}/scripts/bindpcie.sh"

WANDB_PROJECT="sna-proxy-sft-debug"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/${NAME}/logs/"
CHECKPOINT_DIR="${RUN_DIR}/${NAME}/checkpoints/"
DATACACHE_DIR="${RUN_DIR}/${NAME}/data_cache/"
TENSORBOARD_DIR="${RUN_DIR}/${NAME}/tensorboard/"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="/tmp/triton-cache"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "BINDPCIE_SCRIPT=${BINDPCIE_SCRIPT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

SEQ_LEN=32768
TRAIN_SAMPLES=10000
LR_WARMUP_SAMPLES=100
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))
LOG_INTERVAL=1
SAVE_INTERVAL=20
SAVE_RETAIN_INTERVAL=100
GBS=64
LR=1e-5
MIN_LR=2e-6

TOKENIZER_MODEL_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/super_checkpoints/tokenizer"
BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/adithyare/nemotron_ultra/blend_jan21.json"

START_FRESH=${START_FRESH:-1}

# No CUDA graph: baseline for comparing with megaron_sft_proxy_hybridep_bindpcie_cudagraph_attn_mamba.sh
OPTIONS=" \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --distributed-timeout-minutes 30 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --recompute-granularity selective \
        --recompute-modules moe \
        \
        --mtp-use-repeated-layer \
        \
        --context-parallel-size 2 \
        --tensor-model-parallel-size 8 \
        --expert-model-parallel-size 16 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --hybrid-override-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --mtp-hybrid-override-pattern \"*E\" \
        \
        --save-interval ${SAVE_INTERVAL} \
        --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
        --lr $LR \
        --min-lr $MIN_LR \
        --lr-decay-style constant \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval ${LOG_INTERVAL} \
        --micro-batch-size 1 \
        --global-batch-size ${GBS} \
        --overlap-grad-reduce \
        --overlap-param-gather \
        \
        --mtp-num-layers 2 \
        --calculate-per-token-loss \
        --mtp-loss-scaling-factor 0.3 \
        \
        --high-priority-stream-groups ep \
        --manual-gc-interval 10 \
        --ddp-num-buckets 10 \
        --manual-gc \
        \
        --moe-latent-size 2048 \
        --moe-permute-fusion \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --use-fused-weighted-squared-relu \
        \
        --moe-token-dispatcher-type flex \
        --moe-flex-dispatcher-backend hybridep \
        --moe-hybridep-num-sms 32 \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 64 \
        --moe-router-topk 22 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 5.0 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        \
        --attention-backend flash \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        --use-persistent-ckpt-worker \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --exit-duration-in-mins 5750 \
        --no-create-attention-mask-in-dataloader \
        \
        --sequence-parallel \
        --use-distributed-optimizer \
        --override-opt-param-scheduler \
        \
        --mamba-num-heads 256 \
        --is-hybrid-model \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.014 \
        --position-embedding-type none \
        --num-layers 108 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 2 \
        --ffn-hidden-size 5120 \
        --kv-channels 128 \
        --save ${CHECKPOINT_DIR} \
        $([ -z "${START_FRESH}" ] && echo "--load ${CHECKPOINT_DIR}") \
        --per-split-data-args-path ${BLEND_PATH} \
        --data-cache-path ${DATACACHE_DIR} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --no-load-optim \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-timers-to-tensorboard \
        --log-progress \
        --log-energy \
        --log-memory-interval 200 \
        --logging-level 20 \
        --log-straggler \
        --disable-straggler-on-startup \
        --straggler-minmax-count 16 \
        --check-weight-hash-across-dp-replicas-interval 20000 \
        --ddp-pad-buckets-for-high-nccl-busbw \
        --timing-log-option minmax \
        --eval-interval 1000 \
        --eval-iters 14 \
        --te-precision-config-file /lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/code_ultra/te_quant.cfg \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-exp-name ${NAME} \
        --dist-ckpt-strictness log_unexpected \
        --tensorboard-dir ${TENSORBOARD_DIR}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

# Launch via bindpcie: CPU=node, mem=node (NUMA binding per rank). Script reads LOCAL_RANK or SLURM_LOCALID.
LAUNCH_CMD="${BINDPCIE_SCRIPT} --cpu=node --mem=node -- ${RUN_CMD}"

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN,USE_MNNVL,UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS,NVTE_CPU_OFFLOAD_V1,SLURM_LOCALID \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${LAUNCH_CMD}"
