import cv2
from pathlib import Path

image_dir = Path("/mnt/ddn/prod-shared/datasets/DriveLM/nuscenes/samples/CAM_FRONT")
scene_prefix = "n015-2018-07-24-10-42-41+0800"

images = sorted(image_dir.glob(f"{scene_prefix}*.jpg"))

frame = cv2.imread(str(images[0]))
h, w = frame.shape[:2]

video = cv2.VideoWriter(
    "scene001.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    12,      # 12 FPS
    (w, h)
)

for img in images:
    frame = cv2.imread(str(img))
    video.write(frame)

video.release()