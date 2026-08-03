from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader

video_path = "scene001.mp4"

# -----------------------------
# Video information
# -----------------------------
vr = VideoReader(video_path)
fps = vr.get_avg_fps()
num_frames = len(vr)
duration = num_frames / fps

print("=" * 50)
print(f"Frames   : {num_frames}")
print(f"FPS      : {fps:.2f}")
print(f"Duration : {duration:.2f} sec")
print("=" * 50)

# -----------------------------
# Load model
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are an expert in autonomous driving and ego-vehicle trajectory analysis.

The video duration is {duration:.2f} seconds.

Your task is to produce a dense temporal description of the ego vehicle's driving direction over the ENTIRE video.

Analyze ONLY the ego vehicle.
Ignore all surrounding vehicles, pedestrians, traffic lights, and road objects.

Focus ONLY on the direction of the ego vehicle's movement.

Requirements:
- Analyze the entire video from beginning to end.
- Cover the entire duration without skipping any interval.
- Divide the video into as many temporal events as necessary.
- Create a new event whenever the driving direction changes, even slightly.
- Prefer more events rather than fewer.
- Do NOT summarize long periods.

The only valid motion labels are:
- straight
- slight left curve
- slight right curve
- left curve
- right curve

Guidelines:
- Output "straight" whenever the vehicle is moving approximately straight.
- Output "slight left curve" or "slight right curve" for gentle steering.
- Output "left curve" or "right curve" for obvious turns or sustained curves.
- Even small steering corrections should be reported as separate events.
- Ignore acceleration, deceleration, braking, stopping, lane changes, and speed changes.
- Ignore the reason for the maneuver.
- Describe ONLY the vehicle's direction of travel.

Timestamp requirements:
- The first event MUST start at 0.0 seconds.
- The final event MUST end at {duration:.2f} seconds.
- Every timestamp MUST be between 0.0 and {duration:.2f} seconds.
- Consecutive events must be continuous without gaps or overlaps.

Output format:

[start_time - end_time]

Direction:
(straight / slight left curve / slight right curve / left curve / right curve)

Explanation:
Briefly describe only how the driving direction changes during this interval.

Generate as many events as necessary until the end of the video.
Never stop early.
Never summarize multiple direction changes into one event.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

# -----------------------------
# Preprocess
# -----------------------------
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

print("pixel_values_videos:", inputs["pixel_values_videos"].shape)

if "video_grid_thw" in inputs:
    print("video_grid_thw:", inputs["video_grid_thw"])

print("video_inputs:", video_inputs)

# -----------------------------
# Inference
# -----------------------------
generated_ids = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=False,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n========== RESULT ==========\n")
print(output[0])