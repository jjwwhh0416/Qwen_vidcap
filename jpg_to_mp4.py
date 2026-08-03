import cv2
from pathlib import Path

image_dir = Path("/mnt/ddn/prod-shared/datasets/DriveLM/nuscenes/samples/CAM_FRONT")
scene_prefix = "n015-2018-08-01-17-04-15+0800"

# 모든 이미지 불러오기
images = sorted(image_dir.glob(f"{scene_prefix}*.jpg"))

# ===========================
# 시작/끝 파일 이름 지정
# (확장자까지 포함)
# ===========================
start_name = "n015-2018-08-01-17-04-15+0800__CAM_FRONT__1533114306662460.jpg"
end_name   = "n015-2018-08-01-17-04-15+0800__CAM_FRONT__1533114345662460.jpg"

# 시작/끝 인덱스 찾기
start_idx = next(i for i, p in enumerate(images) if p.name == start_name)
end_idx   = next(i for i, p in enumerate(images) if p.name == end_name)

# 시작~끝 이미지 선택
selected_images = images[start_idx:end_idx + 1]

print(f"Selected {len(selected_images)} frames")

# 첫 프레임으로 영상 크기 확인
frame = cv2.imread(str(selected_images[0]))
h, w = frame.shape[:2]

video = cv2.VideoWriter(
    "scene001.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    12,     # FPS
    (w, h)
)

for img in selected_images:
    frame = cv2.imread(str(img))
    video.write(frame)

video.release()

print("Done!")