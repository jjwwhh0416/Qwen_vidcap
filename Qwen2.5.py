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
You are an expert in autonomous driving, vehicle dynamics, and ego-vehicle motion analysis.

The video duration is {duration:.2f} seconds.

Your task is to produce a dense temporal description of the ego vehicle's motion over the ENTIRE video.

Analyze ONLY the ego vehicle.
Ignore surrounding vehicles unless they directly cause the ego vehicle to change its motion.

Requirements:
- Analyze the video from beginning to end.
- Cover the ENTIRE video without skipping any interval.
- Divide the video into as many temporal events as necessary.
- Prefer MORE events rather than fewer events.
- Do NOT merge multiple driving behaviors into a single event.
- Every noticeable change in the ego vehicle's motion should start a new event.
- Even subtle changes should be reported.

Treat the following as separate events whenever they occur:
- slight left steering
- slight right steering
- steering correction
- beginning of a turn
- middle of a turn
- end of a turn
- entering a curve
- exiting a curve
- lane centering adjustment
- lane change
- acceleration
- gradual acceleration
- strong acceleration
- deceleration
- gradual deceleration
- braking
- stop
- waiting
- parked
- starting from rest
- maintaining constant speed

Pay special attention to:
- small steering corrections
- gentle curves
- slight lane-centering adjustments
- subtle speed changes
- transitions between acceleration and constant speed
- transitions between steering directions

Do NOT summarize long periods of driving.
Instead, split them whenever the vehicle's steering angle, speed, or trajectory changes, even slightly.

Requirements for timestamps:
- The first event MUST start at 0.0 seconds.
- The final event MUST end at {duration:.2f} seconds.
- Every timestamp MUST be between 0.0 and {duration:.2f} seconds.
- Consecutive events must be continuous without gaps or overlaps.

For each event, output:

[start_time - end_time]

Motion:
(A concise motion label.)

Explanation:
(A detailed description of exactly how the ego vehicle moves during this interval and why this interval is different from the previous one.)

Generate as many events as necessary to completely describe the vehicle's motion.
Never stop early.
Never summarize the entire video into only a few events.
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