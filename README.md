# bts-digital-twin

## Cai dat

```
pip install -r requirements.txt
```

## Cau truc data

```
data/public_set/<scene>/train/images/
data/public_set/<scene>/train/sparse/0/
data/public_set/<scene>/test/images/
data/public_set/<scene>/test/test_poses.csv

data/private_set1/<scene>/train/images/
data/private_set1/<scene>/train/sparse/0/
data/private_set1/<scene>/test/test_poses.csv
```

Nếu data không nằm trong thư mục `./data`, đặt biến môi trường `DATA_ROOT` trước khi chạy:

```powershell
$env:DATA_ROOT="D:\duong_dan_toi_data"
```

Thư mục đó phải chứa `public_set/` và/hoặc `private_set1/`. Nếu muốn đổi nơi lưu checkpoint/render, đặt thêm:

```powershell
$env:OUTPUT_ROOT="D:\duong_dan_toi_outputs"
```

Với bộ data round2 có cấu trúc scene trực tiếp như `VAI_NVS_DATA_ROUND2/<scene>/train/...`, đặt:

```powershell
$env:ROUND2_ROOT="D:\ViettelAIRace-2026\VAI_NVS_DATA_ROUND2"
```

Sau đó dùng `--split round2`.

## Train 1 scene

```
python scripts/train.py --scene hcm0031 --split private --config configs/base_config.yaml
```

## Train voi holdout validation

```
python scripts/train.py --scene hcm0031 --split private --config configs/base_config.yaml --holdout-ratio 0.1 --holdout-seed 2026 --no-resume
```

## Train toan bo scene

```
python scripts/train_all_scenes.py --split private --config configs/base_config.yaml
```

## Train round2

```
python scripts/train_all_scenes.py --split round2 --config configs/competitive.yaml --no-resume
```

## Train canh tranh hon

`configs/competitive.yaml` bat SH color degree 3, train 30k iteration va dung cac moc gan voi baseline Graphdeco 3DGS hon. Nen benchmark tren `public_set` truoc khi train private vi thoi gian train/VRAM se cao hon:

```
python scripts/train_all_scenes.py --split private --config configs/competitive.yaml --no-resume
```

## Benchmark 1 scene public truoc khi train private

Script nay train mot scene public voi holdout validation, tinh PSNR/SSIM/LPIPS va competition score noi bo. Mac dinh chon scene public dau tien neu khong truyen `--scene`.

```
python scripts/benchmark_public_scene.py --config configs/competitive.yaml --holdout-ratio 0.1 --no-resume --require-lpips
```

Với round2:

```
python scripts/benchmark_public_scene.py --split round2 --config configs/competitive.yaml --holdout-ratio 0.1 --no-resume --require-lpips --min-score 0.60
```

Nếu benchmark round2 đã có checkpoint 30k nhưng score còn hơi thấp, fine-tune thêm 10k iter bằng LPIPS nhẹ:

```
python scripts/benchmark_public_scene.py --split round2 --config configs/round2_finetune.yaml --holdout-ratio 0.1 --require-lpips --min-score 0.60
```

Nếu muốn train round2 từ đầu với cấu hình chất lượng cao hơn:

```
python scripts/benchmark_public_scene.py --split round2 --config configs/round2_quality.yaml --holdout-ratio 0.1 --no-resume --require-lpips --min-score 0.60
```

Co the dat nguong de fail som neu config chua on:

```
python scripts/benchmark_public_scene.py --config configs/competitive.yaml --holdout-ratio 0.1 --no-resume --require-lpips --min-score 0.55 --min-ssim 0.45 --max-lpips 0.45
```

## Render 1 scene

```
python scripts/render.py --scene hcm0031 --split private
```

## Render toan bo scene

```
python scripts/render_all_scenes.py --split private
```

## Local validation tren holdout

```
python scripts/eval_local.py --scene hcm0031 --split private --holdout-ratio 0.1 --holdout-seed 2026
```

## Dong goi submission

```
python scripts/package_submission.py
```

## Kiem tra submission truoc khi nop

```
python scripts/validate_submission.py --split private
```

## Kiem tra file zip submission

```
python scripts/validate_submission.py --split private --zip outputs/submission_round1.zip
```


## Cải thiện training hiện tại

Pipeline train hiện đã bật một số cơ chế 3DGS quan trọng trong `configs/base_config.yaml`:

- `lr_decay`: giảm learning rate dần về cuối train để fine-tune ổn định hơn.
- `sh_degree`: `0` dùng RGB cố định như baseline; `3` trong `competitive.yaml` bật spherical harmonics để học màu phụ thuộc góc nhìn tốt hơn.
- `densification`: clone thêm Gaussian ở vùng có gradient cao, giúp model tăng chi tiết ở vùng còn lỗi.
- `pruning`: bỏ Gaussian opacity thấp để giảm nhiễu/floater và tiết kiệm VRAM.
- `opacity_reset`: reset opacity định kỳ để Gaussian còn cơ hội học lại phân bố alpha.
- `lambda_lpips`: mặc định tắt, có thể bật nhẹ khi fine-tune để tối ưu cảm quan.

## Cấu trúc thư mục

```text
.
├── README.md                         
├── requirements.txt                  
├── cai_thien_pipeline_3dgs.md        
├── configs/
│   ├── base_config.yaml              # Cấu hình train mặc định cho số vòng lặp, loss và learning rate.
│   ├── competitive.yaml              # Cấu hình train mạnh hơn: SH degree 2, nhiều iteration hơn và densification rộng hơn.
│   └── scene_default.yaml            # Cấu hình mẫu để override riêng cho từng scene khi cần.
├── scripts/
│   ├── train.py                      # Train một scene và lưu checkpoint vào outputs/checkpoints.
│   ├── train_all_scenes.py           # Train lần lượt toàn bộ scene trong public/private split.
│   ├── benchmark_public_scene.py     # Train/evaluate một scene public holdout trước khi chạy private.
│   ├── render.py                     # Render ảnh test cho một scene từ checkpoint mới nhất.
│   ├── render_all_scenes.py          # Render toàn bộ scene đã có checkpoint.
│   ├── eval_local.py                 # Đánh giá nội bộ trên holdout train images bằng PSNR/SSIM/LPIPS/score.
│   ├── package_submission.py         # Đóng gói outputs/rendered thành file zip submission.
│   └── validate_submission.py        # Kiểm tra thiếu/thừa scene, thiếu/thừa ảnh và sai kích thước trước khi nộp.
├── src/
│   ├── config.py                     # Khai báo đường dẫn data/output và helper lấy đường dẫn scene.
│   ├── data/
│   │   ├── dataset.py                # Đọc COLMAP binary, ảnh train, camera train và point cloud.
│   │   ├── pose_loader.py            # Đọc test_poses.csv thành danh sách pose cần render.
│   │   ├── splits.py                 # Tạo holdout split ổn định theo seed cho local validation.
│   │   └── transforms.py             # Chuyển đổi ảnh giữa PIL, tensor và numpy array.
│   ├── models/
│   │   ├── camera.py                 # Tạo ma trận camera intrinsics và view matrix cho renderer.
│   │   ├── gaussian_splatting.py     # Định nghĩa GaussianModel và gọi gsplat để render.
│   │   └── losses.py                 # Chứa L1, SSIM, loss kết hợp và LPIPS loss tùy chọn.
│   ├── training/
│   │   ├── trainer.py                # Vòng lặp train chính: render, tính loss, backprop và save checkpoint.
│   │   └── checkpoint.py             # Lưu, load và tìm checkpoint mới nhất.
│   ├── inference/
│   │   ├── renderer.py               # Load checkpoint và render ảnh theo test pose.
│   │   └── postprocess.py            # Kiểm tra kích thước và lưu ảnh render ra đĩa.
│   ├── evaluation/
│   │   └── metrics.py                # Tính PSNR, SSIM, LPIPS và score validation nội bộ.
│   └── utils/
│       ├── io_utils.py               # Hàm tiện ích đọc/ghi ảnh, YAML, JSON và tạo thư mục.
│       └── logger.py                 # Tạo logger thống nhất cho các script.
└── outputs/
    ├── checkpoints/.gitkeep          
    └── rendered/.gitkeep            
```

