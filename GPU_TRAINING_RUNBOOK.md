# Hướng Dẫn Train Trên Máy GPU Thuê

Tài liệu này dùng cho lần sau khi thuê máy GPU mới để train repo `ViettelAIRace-2026`.

Mục tiêu:

- Không tốn cả giờ GPU chỉ để copy data bằng `scp -r`.
- Benchmark trên `public_set` trước khi train `private_set1`.
- Có thể resume nếu mất kết nối.
- Validate đầy đủ trước khi đóng gói file submission.

## 0. Cách Nhanh Hơn `scp -r`

`scp -r` chậm vì dataset có nhiều file ảnh nhỏ. Cách nên dùng nhất là nén data thành một file lớn rồi upload.

### Cách Khuyên Dùng: Nén Data Thành Một File

Trên máy local Windows, mở Git Bash:

```bash
cd /d/ViettelAIRace-2026
tar -czf data_viettel.tar.gz data/public_set data/private_set1
```

Upload một file lên server GPU:

```bash
scp -P PORT data_viettel.tar.gz root@IP:~/viettel_airace/
```

Trên server GPU, giải nén:

```bash
cd ~/viettel_airace
tar -xzf data_viettel.tar.gz
ls -lah ~/viettel_airace/data
```

Cần thấy:

```text
public_set
private_set1
```

### Cách Có Resume: Dùng `rsync`

Nếu Git Bash có `rsync`, có thể dùng cách này để copy tiếp phần thiếu nếu bị ngắt mạng:

```bash
rsync -avh --progress --partial -e "ssh -p PORT" /d/ViettelAIRace-2026/data/public_set root@IP:~/viettel_airace/data/
rsync -avh --progress --partial -e "ssh -p PORT" /d/ViettelAIRace-2026/data/private_set1 root@IP:~/viettel_airace/data/
```

Nếu bị mất kết nối, chạy lại đúng lệnh cũ. `rsync` sẽ chỉ copy phần còn thiếu.

### Cách Tốt Nhất Nếu Có Persistent Volume

Nếu nhà cung cấp GPU cho attach volume riêng:

1. Tạo volume lưu data, ví dụ `/workspace/data`.
2. Upload hoặc tải data vào volume một lần.
3. Lần sau thuê máy mới thì attach lại volume đó.
4. Không cần copy data lại từ máy local.

Đây là cách tiết kiệm tiền GPU nhất nếu bạn phải thuê máy nhiều lần.

## 1. SSH Vào Máy GPU

Dùng lệnh nhà cung cấp đưa. Ví dụ:

```bash
ssh -p PORT root@IP
```

Nếu gặp lỗi host key changed:

```bash
ssh-keygen -R "[IP]:PORT"
ssh -p PORT root@IP
```

Kiểm tra GPU:

```bash
nvidia-smi
```

Với RTX 3090, nên thấy GPU 24GB VRAM.

## 2. Chuẩn Bị Thư Mục

Trên server:

```bash
mkdir -p ~/viettel_airace/data ~/viettel_airace/outputs
```

Sau khi upload hoặc giải nén data, kiểm tra:

```bash
ls -lah ~/viettel_airace/data
ls ~/viettel_airace/data/public_set | head
ls ~/viettel_airace/data/private_set1 | head
```

## 3. Cài Tool Hệ Thống

```bash
apt-get update
apt-get install -y git build-essential ninja-build tmux
```

## 4. Clone Repo

```bash
cd ~
git clone https://github.com/nptuyenn/ViettelAIRace-2026.git
cd ViettelAIRace-2026
git pull origin main
```

Nếu repo đã tồn tại:

```bash
cd ~/ViettelAIRace-2026
git pull origin main
```

## 5. Cài Python Package Cho RTX 3090

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -q pyyaml lpips tqdm pillow numpy ninja packaging gsplat
```

Kiểm tra:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

Cần thấy:

```text
cuda True
RTX 3090
capability (8, 6)
```

## 6. Chạy Trong `tmux`

Dùng `tmux` để job train không chết khi mất SSH:

```bash
tmux new -s viettel
```

Trong `tmux`:

```bash
cd ~/ViettelAIRace-2026
export DATA_ROOT=~/viettel_airace/data
export OUTPUT_ROOT=~/viettel_airace/outputs
```

Thoát khỏi `tmux` mà job vẫn chạy:

```text
Ctrl+B, rồi bấm D
```

Vào lại:

```bash
tmux attach -t viettel
```

## 7. Benchmark Public Trước

Chạy một scene public với holdout validation. Điểm leaderboard 60/100 tương đương `score_mean >= 0.60`.

```bash
python scripts/benchmark_public_scene.py \
  --config configs/competitive.yaml \
  --holdout-ratio 0.1 \
  --no-resume \
  --require-lpips \
  --min-score 0.60
```

Xem report:

```bash
cat ~/viettel_airace/outputs/public_benchmark.json
```

Nếu score dưới `0.60`, khoan train private. Cần tune config/model tiếp.

## 8. Train Private

Chỉ chạy sau khi public benchmark ổn.

```bash
python scripts/train_all_scenes.py \
  --split private \
  --config configs/competitive.yaml \
  --no-resume
```

Nếu bị ngắt giữa chừng, resume bằng cách bỏ `--no-resume`:

```bash
python scripts/train_all_scenes.py \
  --split private \
  --config configs/competitive.yaml
```

## 9. Render Private

```bash
python scripts/render_all_scenes.py --split private
```

## 10. Validate Rendered Folder

```bash
python scripts/validate_submission.py \
  --split private \
  --rendered_dir ~/viettel_airace/outputs/rendered
```

Nếu báo thiếu file ở scene nào, render lại scene đó:

```bash
python scripts/render.py --scene SCENE_NAME --split private
```

## 11. Package Zip

```bash
python scripts/package_submission.py \
  --rendered_dir ~/viettel_airace/outputs/rendered \
  --output ~/viettel_airace/outputs/submission_round1.zip
```

Validate zip:

```bash
python scripts/validate_submission.py \
  --split private \
  --zip ~/viettel_airace/outputs/submission_round1.zip
```

File nộp:

```text
~/viettel_airace/outputs/submission_round1.zip
```

## 12. Tải Zip Về Local

Từ terminal local:

```bash
scp -P PORT root@IP:~/viettel_airace/outputs/submission_round1.zip .
```

Hoặc dùng File Browser của nhà cung cấp nếu có.

## 13. Checklist Rút Gọn

```text
1. SSH vào máy mới.
2. Chạy nvidia-smi.
3. Tạo ~/viettel_airace/data và ~/viettel_airace/outputs.
4. Lấy data bằng archive/rsync/persistent volume.
5. Clone repo và git pull.
6. Cài torch + dependencies.
7. Chạy tmux new -s viettel.
8. Export DATA_ROOT và OUTPUT_ROOT.
9. Chạy benchmark_public_scene.py --min-score 0.60.
10. Nếu pass: train_all_scenes private.
11. Render private.
12. Validate rendered.
13. Package submission.
14. Validate zip.
15. Download submission_round1.zip.
```
