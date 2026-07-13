# Kế hoạch cải thiện pipeline 3D Gaussian Splatting

> Mục tiêu bài toán: từ tập ảnh drone đa góc nhìn của trạm BTS + camera pose (COLMAP), tái dựng cấu trúc 3D ngầm và **sinh ảnh RGB tại các góc nhìn mới** sao cho giống ảnh ground-truth nhất.
>
> Công thức chấm: `Score = 0.4×(1−LPIPS) + 0.3×SSIM + 0.3×PSNR_norm`
> → **LPIPS chiếm trọng số lớn nhất (40%)**, nên chất lượng cảm quan quan trọng hơn sai số pixel thuần.

Cấu trúc thư mục hiện tại đã bao phủ đúng các thành phần của pipeline (data → model → training → inference → evaluation → packaging). Vấn đề không nằm ở việc thiếu thư mục, mà ở các chi tiết bên trong quyết định điểm số. Tài liệu này sắp xếp cải tiến theo mức độ ảnh hưởng đến `Score`.

---

## Nhóm 1 — Các "bẫy" âm thầm giết điểm (ưu tiên cao nhất)

Đây là những lỗi khiến ảnh render bị lệch **hệ thống**, kéo tụt cả ba metric mà rất khó phát hiện nếu chỉ nhìn ảnh bằng mắt.

### 1.1. Principal point (cx, cy)
- Rasterizer gốc của INRIA giả định principal point nằm **chính giữa** ảnh và chỉ nhận FoV.
- Nhưng `test_poses.csv` cung cấp `cx, cy` tường minh, và với ảnh drone/COLMAP chúng thường **không** ở tâm.
- Nếu bỏ qua → toàn bộ ảnh render bị dịch một cách hệ thống, hỏng điểm dù nhìn thoáng qua vẫn thấy "đúng đối tượng".
- **Xử lý:** dùng rasterizer hỗ trợ PP offset (ví dụ `gsplat` của nerfstudio), hoặc render ở khung lớn hơn rồi crop theo `cx, cy`.
- 👉 **Nên kiểm tra đầu tiên.**

### 1.2. Intrinsics & kích thước theo từng pose
- `fx` và `fy` có thể khác nhau (focal bất đẳng hướng) → phải xử lý riêng từng trục.
- Mỗi pose có `width/height` riêng → renderer phải render **đúng độ phân giải yêu cầu**.
- ❌ Không được render một size rồi resize — resize làm hỏng LPIPS/SSIM.

### 1.3. Convention tọa độ COLMAP
- Quaternion + translation trong COLMAP là **world-to-camera** (không phải camera-to-world).
- Hệ tọa độ khác với OpenGL.
- `pose_loader.py` phải convert đúng, nếu không mọi thứ sai từ gốc.
- **Nên có** `src/data/colmap_utils.py` đọc `.bin` tường minh, kèm unit test đối chiếu lại với ảnh train.

### 1.4. Siết chặt `validate_submission.py`
Phải đối chiếu trực tiếp với `test_poses.csv`:
- Đúng tên scene
- Đúng tên file ảnh
- Đúng số lượng ảnh mỗi scene
- Đúng `width × height` từng ảnh

> ⚠️ **Luật:** thiếu hoặc thừa scene → **cả bài không được tính điểm.**

---

## Nhóm 2 — Tối ưu trực tiếp theo công thức chấm

### 2.1. Thêm LPIPS loss (quan trọng — nhắm vào 40% điểm)
- Loss mặc định của 3DGS là `L1 + D-SSIM`, **không hề đụng tới LPIPS**.
- Thêm một **giai đoạn fine-tune với LPIPS loss** (hoặc trộn perceptual loss trọng số nhỏ từ đầu) trong `losses.py`.
- Đây là cách tối ưu trực tiếp cho thành phần chiếm trọng số lớn nhất.

### 2.2. Anti-aliasing kiểu Mip-Splatting
- Nếu intrinsics/độ phân giải test khác train (rất hay xảy ra) → 3DGS gốc bị aliasing và erosion, hỏng cả ba metric.
- Đây là nâng cấp gần như "miễn phí" về chất lượng.

### 2.3. Depth regularization
- Dùng chính `points3D.bin` làm sparse depth prior (hoặc mono-depth).
- Giúp giảm floater ở các góc nhìn novel — đặc biệt quan trọng khi test pose là **extrapolation** ra ngoài quỹ đạo drone, nơi 3DGS hay sinh vệt mờ.

---

## Nhóm 3 — Thứ đang thiếu trong cấu trúc

### 3.1. Local validation harness ⭐ (quan trọng nhất về vòng lặp làm việc)
- Hiện chưa có cách đo điểm **trước khi nộp**.
- Hold-out một phần ảnh train làm validation nội bộ.
- Viết `scripts/eval_local.py` tính chính xác công thức `Score` (dùng `metrics.py`).
- Không có nó → tối ưu mù. Có nó → mỗi thay đổi đều đo được ngay.

### 3.2. Appearance / exposure modeling
- Drone bay quanh trạm BTS thường bật auto-exposure / white-balance → độ sáng ảnh không nhất quán.
- Một **per-image appearance embedding** (kiểu NeRF-W / GS-W) giúp model không "học nhầm" biến thiên phơi sáng thành hình học → cải thiện đáng kể LPIPS.

### 3.3. Masking cho vật thể động / nền trời
- Người qua lại, cây rung, bầu trời phẳng → nên có mask để loại khỏi loss, tránh floater.

### 3.4. Per-scene config override
- Scene 100 ảnh và scene 300 ảnh cần số iteration / ngưỡng densification khác nhau.
- Cho phép override qua config; lý tưởng là **tự chọn** theo số ảnh / mật độ point cloud.

---

## Nhóm 4 — Reproducibility (luật bắt buộc, Điều 10.3)

Top đội **phải** nộp được để chứng minh tái lập kết quả:
- Mã nguồn huấn luyện + suy luận
- File cấu hình (config)
- Danh sách thư viện + **phiên bản cụ thể**
- Checkpoint mô hình
- Nhật ký huấn luyện (training logs)

**Chuẩn bị từ đầu thay vì chữa cháy:**
- [ ] Thêm `requirements.txt` / `environment.yml` **pin phiên bản** (CUDA/torch của 3DGS rất nhạy).
- [ ] Set seed cố định + chế độ deterministic.
- [ ] Đảm bảo `logger.py` lưu log ra file trong `outputs/`.
- [ ] Thêm `README.md` mô tả cách chạy end-to-end.

---

## Tổng hợp: các file / việc cần thêm

| File / Việc | Mục đích | Nhóm |
|---|---|---|
| `src/data/colmap_utils.py` | Đọc `.bin` + convert convention đúng | 1.3 |
| Xử lý PP offset trong `renderer.py` / `camera.py` | Sửa lệch cx, cy | 1.1 |
| Siết `validate_submission.py` | Đối chiếu với `test_poses.csv` | 1.4 |
| LPIPS loss trong `losses.py` | Tối ưu 40% điểm | 2.1 |
| `scripts/eval_local.py` | Holdout + tính Score nội bộ | 3.1 |
| `requirements.txt` (pin version) | Reproducibility | 4 |
| `README.md` | Hướng dẫn chạy end-to-end | 4 |

---

## Thứ tự thực hiện đề xuất

1. **Trước tiên:** dựng `eval_local.py` + holdout validation → để đo được mọi thay đổi sau đó.
2. Sửa 3 bẫy hình học: PP offset, intrinsics per-pose, convention COLMAP.
3. Siết `validate_submission.py`.
4. Thêm LPIPS loss + anti-aliasing.
5. Depth regularization + appearance embedding.
6. Hoàn thiện reproducibility (requirements, seed, log, README).
