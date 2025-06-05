BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_metamath.py \
    --data-dir /data/zitai/guanghui/distillm-master/data/metamath/ \
    --processed-data-dir ${BASE_PATH}/processed_data/metamath/full \
    --model-path qwen2.5-math-7b \
    --data-process-workers 16 \
    --max-prompt-length 256 \
    --dev-num 5000 \
    --model-type qwen \
    --data-length 55000
