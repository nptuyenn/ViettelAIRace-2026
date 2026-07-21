# Hướng Dẫn Train Trên Máy GPU Thuê

Tài liệu này ưu tiên tiết kiệm thời gian thuê GPU. Việc nén data nên làm **trước khi khởi động máy GPU**.

## 0. Làm Trước Khi Bật GPU

Trên máy local Windows, mở Git Bash:

```bash
cd /d/ViettelAIRace-2026
git pull origin main
tar -czf round2_data.tar.gz VAI_NVS_DATA_ROUND2
ls -lh round2_data.tar.gz
```

Giữ file `round2_data.tar.gz` lại. Lần sau nếu data round2 không đổi thì không cần nén lại.

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
scp -P PORT /d/ViettelAIRace-2026/round2_data.tar.gz root@IP:~/viettel_airace/
```

## 2. Setup Server GPU

Chạy block này trên server sau khi SSH vào:

```bash
mkdir -p ~/viettel_airace/outputs
cd ~/viettel_airace
tar -xzf round2_data.tar.gz

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
ls -lah ~/viettel_airace/VAI_NVS_DATA_ROUND2
ls ~/viettel_airace/VAI_NVS_DATA_ROUND2 | head
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
  --scene HCM0421 \
  --split round2 \
  --config configs/round2_geometry.yaml \
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

`round2_geometry.yaml` train từ đầu với clone/split densification, tích lũy AbsGrad và antialiased rasterization. Đây là nhánh nên thử sau khi cấu hình 40k/50k cũ bị kẹt quanh `0.57-0.58`.

Nếu benchmark geometry đã gần đạt, ví dụ khoảng `0.58-0.60`, có thể fine-tune tiếp bằng loss cảm quan nhẹ:

```bash
python3 scripts/benchmark_public_scene.py \
  --scene HCM0421 \
  --split round2 \
  --config configs/round2_quality.yaml \
  --holdout-ratio 0.1 \
  --require-lpips \
  --min-score 0.60
```

Lệnh fine-tune trên cố ý không có `--no-resume`, để tiếp tục từ checkpoint geometry hiện có. Không nên dùng `round2_metric_push.yaml` làm đường chính nữa vì lần thử HCM0421 của bạn đã tụt từ khoảng `0.579` xuống `0.550`.

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
  --config configs/round2_quality.yaml \
  --no-resume
```

Nếu bị ngắt giữa chừng, resume bằng cách bỏ `--no-resume`:

```bash
python3 scripts/train_all_scenes.py \
  --split round2 \
  --config configs/round2_quality.yaml
```

## 5. Render, Validate, Package

```bash
python3 scripts/render_all_scenes.py --split round2

python3 scripts/validate_submission.py \
  --split round2 \
  --rendered_dir ~/viettel_airace/outputs/rendered

python3 scripts/package_submission.py \
  --rendered_dir ~/viettel_airace/outputs/rendered \
  --output ~/viettel_airace/outputs/submission_round2.zip

python3 scripts/validate_submission.py \
  --split round2 \
  --zip ~/viettel_airace/outputs/submission_round2.zip
```

File nộp:

```text
~/viettel_airace/outputs/submission_round2.zip
```

Tải về local:

```bash
scp -P PORT root@IP:~/viettel_airace/outputs/submission_round2.zip .
```

## Ghi Chú Nhanh

- Đừng dùng `scp -r` cho từng ảnh nếu không bắt buộc. Hãy upload `round2_data.tar.gz`.
- Nếu nhà cung cấp có persistent volume, hãy lưu data ở volume đó để lần sau không cần upload lại.
- RTX 3090 dùng `TORCH_CUDA_ARCH_LIST="8.6"`.
- Nếu build `gsplat` báo thiếu `Python.h`, cài `python3-dev python3.10-dev`.
