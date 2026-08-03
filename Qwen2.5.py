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
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

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
You are an expert in autonomous driving and ego-vehicle motion analysis.

The video duration is {duration:.2f} seconds.

Analyze the ENTIRE video from beginning to end.

Describe ONLY the motion of the ego vehicle.
Ignore surrounding vehicles unless they directly affect the ego vehicle.

Requirements:
- Cover the entire video.
- Split the video into chronological temporal events.
- Continue until the end of the video.
- The first event MUST start at 0.0 seconds.
- The final event MUST end at {duration:.2f} seconds.
- Every timestamp MUST be between 0.0 and {duration:.2f} seconds.
- Consecutive events must not overlap.
- Do not omit any part of the video.

Motion labels include:
- straight
- slight left steering
- slight right steering
- left turn
- right turn
- lane change left
- lane change right
- acceleration
- deceleration
- stop
- waiting
- parked

Output format:

[start_time - end_time]
Motion:
Explanation:
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