# Hướng Dẫn Train Trên Máy GPU Thuê

Tài liệu này ưu tiên tiết kiệm thời gian thuê GPU. Việc nén data nên làm **trước khi khởi động máy GPU**.

## 0. Làm Trước Khi Bật GPU

Trên máy local Windows, mở Git Bash:

```bash
cd /d/ViettelAIRace-2026
git pull origin main
tar -czf data_viettel.tar.gz VAI_NVS_DATA_ROUND2 data/public_set data/private_set1
ls -lh data_viettel.tar.gz
```

Giữ file `data_viettel.tar.gz` lại. Lần sau nếu data không đổi thì không cần nén lại.

## 1. Khi Có Máy GPU Mới

Thay `PORT` và `IP` bằng thông tin nhà cung cấp đưa.

Từ máy local, SSH thử:

```bash
ssh -p PORT root@IP
```

Nếu gặp lỗi host key changed:

```bash
ssh-keygen -R "[IP]:PORT"
ssh -p PORT root@IP
```

Upload data archive từ máy local:

```bash
scp -P PORT /d/ViettelAIRace-2026/data_viettel.tar.gz root@IP:~/viettel_airace/
```

## 2. Setup Server GPU

Chạy block này trên server sau khi SSH vào:

```bash
mkdir -p ~/viettel_airace/outputs
cd ~/viettel_airace
tar -xzf data_viettel.tar.gz

apt-get update
apt-get install -y git build-essential ninja-build tmux python3-dev python3.10-dev

cd ~
git clone https://github.com/nptuyenn/ViettelAIRace-2026.git || true
cd ~/ViettelAIRace-2026
git pull origin main

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -q pyyaml lpips tqdm pillow numpy ninja packaging jaxtyping rich

export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.6"
pip3 install -v --no-build-isolation gsplat
```

Kiểm tra môi trường:

```bash
python3 - <<'PY'
import torch
import gsplat
from gsplat import rasterization
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("gsplat OK")
PY
```

Kiểm tra data:

```bash
ls -lah ~/viettel_airace/data
ls ~/viettel_airace/data/public_set | head
ls ~/viettel_airace/data/private_set1 | head
```

## 3. Chạy Benchmark Public Trong `tmux`

```bash
tmux new -s viettel
```

Trong `tmux`:

```bash
cd ~/ViettelAIRace-2026
export DATA_ROOT=~/viettel_airace/data
export ROUND2_ROOT=~/viettel_airace/VAI_NVS_DATA_ROUND2
export OUTPUT_ROOT=~/viettel_airace/outputs
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.6"

rm -rf ~/viettel_airace/outputs/public_benchmark_checkpoints

python3 scripts/benchmark_public_scene.py \
  --split round2 \
  --config configs/competitive.yaml \
  --holdout-ratio 0.1 \
  --no-resume \
  --require-lpips \
  --min-score 0.60
```

Xem điểm:

```bash
cat ~/viettel_airace/outputs/public_benchmark.json
```

Nếu `score_mean < 0.60`, khoan train private.

Thoát khỏi `tmux` mà job vẫn chạy:

```text
Ctrl+B, rồi bấm D
```

Vào lại:

```bash
tmux attach -t viettel
```

## 4. Train Private Nếu Benchmark Ổn

```bash
cd ~/ViettelAIRace-2026
export DATA_ROOT=~/viettel_airace/data
export ROUND2_ROOT=~/viettel_airace/VAI_NVS_DATA_ROUND2
export OUTPUT_ROOT=~/viettel_airace/outputs

python3 scripts/train_all_scenes.py \
  --split round2 \
  --config configs/competitive.yaml \
  --no-resume
```

Nếu bị ngắt giữa chừng, resume bằng cách bỏ `--no-resume`:

```bash
python3 scripts/train_all_scenes.py \
  --split round2 \
  --config configs/competitive.yaml
```

## 5. Render, Validate, Package

```bash
python3 scripts/render_all_scenes.py --split round2

python3 scripts/validate_submission.py \
  --split round2 \
  --rendered_dir ~/viettel_airace/outputs/rendered

python3 scripts/package_submission.py \
  --rendered_dir ~/viettel_airace/outputs/rendered \
  --output ~/viettel_airace/outputs/submission_round1.zip

python3 scripts/validate_submission.py \
  --split round2 \
  --zip ~/viettel_airace/outputs/submission_round1.zip
```

File nộp:

```text
~/viettel_airace/outputs/submission_round1.zip
```

Tải về local:

```bash
scp -P PORT root@IP:~/viettel_airace/outputs/submission_round1.zip .
```

## Ghi Chú Nhanh

- Đừng dùng `scp -r` cho từng ảnh nếu không bắt buộc. Hãy upload `data_viettel.tar.gz`.
- Nếu nhà cung cấp có persistent volume, hãy lưu data ở volume đó để lần sau không cần upload lại.
- RTX 3090 dùng `TORCH_CUDA_ARCH_LIST="8.6"`.
- Nếu build `gsplat` báo thiếu `Python.h`, cài `python3-dev python3.10-dev`.
