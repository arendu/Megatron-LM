#!/bin/bash
#
# 64-node SFT — baseline (recompute + offload만 적용), EP=128, CP=16
#
# megaron_sft_64node_baseline.sh 에서 변경:
#   - EP=64 → EP=128, CP=32 → CP=16 (HybridEP 없이 alltoall)
#   - async-save 제거
#
# 64-node: nodes=64, segment=16, TP=8, EP=128, CP=16, experts=512, seq=262144

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=llmservice_nemotron_ultra
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=64
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --segment=16
#SBATCH --job-name=ultra-sft-64node-baseline-ep128-cp16

################################################################
### TransformerEngine (SM margin만 유지)
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
export NVTE_CPU_OFFLOAD_V1=1

################################################################
### General (최소한만)
################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-LM/sft-runs"
MEGATRON_LM_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-LM"
IMAGE="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/containers/pt_ultra_mamba_ssmv230_23jan28.sqsh"

WANDB_PROJECT="ultra-v3-sft-hsg-hybridep"

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

BASE_MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_ultra/reinit-embeddings-ckpts/phase1_fp32rs_continued_iter_0600000/checkpoints"

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
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

SEQ_LEN=262144
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
        --fine-grained-activation-offloading \
        --offload-modules moe_act \
        \
        --mtp-use-repeated-layer \
        \
        --context-parallel-size 16 \
        --tensor-model-parallel-size 8 \
        --expert-model-parallel-size 128 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --hybrid-override-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --mtp-hybrid-override-pattern \"*E\" \
        \
        --pretrained-checkpoint ${BASE_MODEL_PATH} \
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
        --moe-latent-size 2048 \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 512 \
        --moe-router-topk 22 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 5.0 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        --moe-token-dispatcher-type alltoall \
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
        --load ${CHECKPOINT_DIR} \
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
        --timing-log-option minmax \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-exp-name ${NAME} \
        --dist-ckpt-strictness log_unexpected \
        --tensorboard-dir ${TENSORBOARD_DIR}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=NVTE_CPU_OFFLOAD_V1,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,TORCHINDUCTOR_WORKER_START,PYTORCH_CUDA_ALLOC_CONF,SLURM_LOCALID \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${RUN_CMD}"
