#!/usr/bin/env bash
set -e  # ⭐ 任意一步失败立即退出

# =========================
# 基本配置
# =========================
export MASTER_PORT=1091

user_dir=/root/BiomedGPT/module
bpe_dir=/root/BiomedGPT/utils/BPE 

image_path=/root/BiomedGPT/examples/242048139049405.jpg
tsv_file=/root/BiomedGPT/examples/inference_input.tsv
model_path=/root/BiomedGPT/checkpoints/biomedgpt_tiny.pt
result_path=./results/single_image

# =========================
# 环境检查（建议保留）
# =========================
echo "Checking environment..."

if ! command -v torchrun &> /dev/null
then
    echo "ERROR: torchrun not found. Please install PyTorch correctly."
    exit 1
fi

if [ ! -f "$image_path" ]; then
    echo "ERROR: Image not found at $image_path"
    exit 1
fi

if [ ! -f "$model_path" ]; then
    echo "ERROR: Model checkpoint not found at $model_path"
    exit 1
fi

# =========================
# 生成 TSV 输入
# =========================
echo "Preparing TSV input..."

python3 - << EOF
import base64
from io import BytesIO
from PIL import Image

image_path = "$image_path"
tsv_file = "$tsv_file"

with open(image_path, 'rb') as f:
    image = Image.open(f).convert("RGB")  # ⭐ 保证格式正确
    img_buffer = BytesIO()
    image.save(img_buffer, format="JPEG")  # ⭐ 避免 format=None 报错
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")

with open(tsv_file, 'w') as out:
    out.write('1\t242048139049405\t\t\t' + base64_str + '\n')

print("TSV file created:", tsv_file)
EOF

# =========================
# 创建输出目录
# =========================
mkdir -p $result_path

# =========================
# 推理
# =========================
echo "Running inference..."

export PYTHONPATH=/root/BiomedGPT/fairseq:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 ./evaluate.py $tsv_file \
    --path=$model_path \
    --user-dir=$user_dir \
    --task=caption \
    --batch-size=1 \
    --log-format=simple \
    --log-interval=10 \
    --seed=7 \
    --gen-subset=test \
    --results-path=$result_path \
    --beam=2 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"$tsv_file\",\"bpe_dir\":\"$bpe_dir\",\"eval_cider\":False,\"selected_cols\":\"1,4,2\"}"

# =========================
# 输出结果
# =========================
RESULT_FILE=${result_path}/test_predict.json

if [ -f "$RESULT_FILE" ]; then
    echo "Inference completed. Results:"
    cat $RESULT_FILE
else
    echo "ERROR: Inference failed. Result file not found!"
    exit 1
fi