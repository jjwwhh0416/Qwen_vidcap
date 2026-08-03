from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are an expert in autonomous driving, vehicle dynamics, and ego-vehicle trajectory analysis.

The video duration is {duration:.2f} seconds.

Your task is to analyze the ego vehicle's driving direction throughout the entire video.

Analyze the video from beginning to end and produce a dense temporal description.

Requirements:
- Cover the entire video.
- Divide the video into chronological temporal events.
- Create a new event whenever the driving direction changes, even slightly.
- Prefer more events rather than fewer.
- Do not merge multiple direction changes into a single event.
- Continue generating events until the end of the video.

Pay attention to:
- gentle road curvature
- steering corrections
- entering a curve
- exiting a curve
- transitions between straight driving and turning
- sustained left curves
- sustained right curves
- subtle changes in heading

Use ONLY the following direction labels:
- straight
- left curve
- right curve

If the vehicle follows a left-curving road, output "left curve".
If the vehicle follows a right-curving road, output "right curve".
Otherwise, output "straight".

Timestamp requirements:
- The first event must start at 0.0 seconds.
- The final event must end at {duration:.2f} seconds.
- Every timestamp must lie between 0.0 and {duration:.2f} seconds.
- Consecutive events must be continuous with no gaps or overlaps.

Output format:

[start_time - end_time]

Direction:
(straight / left curve / right curve)

Explanation:
Briefly describe how the driving direction changes during this interval.

Generate as many events as necessary to completely describe the driving direction.
Never stop early.
Do not summarize long intervals into a single event if the direction changes.
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