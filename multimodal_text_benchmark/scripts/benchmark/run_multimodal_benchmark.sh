set -x

DATE_WITH_TIME=`date "+%Y-%m-%d_%H-%M-%S"`
MODEL_TYPE=${1:-ag_tabular_old}
TABULAR_PRESETS=${2:-no}
TEXT_PRESETS=${3:-no}
SEED=${4:-123}
GPU_ID=${5:-0}
DATASET_FILE=${6:-./benchmark_datasets.txt}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

for dataset in $(cat ${DATASET_FILE})
do
    SAVE_PATH=${MODEL_TYPE}-${TEXT_PRESETS}-${TABULAR_PRESETS}/${dataset}/${DATE_WITH_TIME}-SEED${SEED}
    mkdir -p ${SAVE_PATH}
    python3 ag_benchmark.py --dataset ${dataset} \
                            --model ${MODEL_TYPE} \
                            --text_presets ${TEXT_PRESETS} \
                            --tabular_presets ${TABULAR_PRESETS}\
                            --seed ${SEED} \
                            --save_dir ${SAVE_PATH}  2>&1 | tee -a ${SAVE_PATH}/log.txt
done
