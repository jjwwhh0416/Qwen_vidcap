from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from decord import VideoReader

video_path = "scene001.mp4"

vr = VideoReader(video_path)
duration = len(vr) / 12

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "scene001.mp4",
            },
            {
                "type": "text",
                "text":
                f"""
You are an expert in autonomous driving and ego-vehicle motion analysis.

The video duration is {duration:.2f} seconds.
Your task is to analyze the ENTIRE video from beginning to end.

Describe ONLY the motion of the ego vehicle.
Ignore surrounding vehicles unless they directly affect the ego vehicle's behavior.

The video may be long.
DO NOT summarize only the beginning of the video.
You MUST analyze the complete video from start to finish.

Requirements:
- Cover the entire video.
- Split the video into consecutive temporal events.
- Continue generating events until the end of the video.
- Do not stop after only a few events.
- Do not omit any portion of the video.

For each event, output:

[start_time - end_time]
Motion:
Explanation:

Motion should be one or more of:
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

Use timestamps in seconds.
If the exact timestamp is uncertain, estimate it as accurately as possible.

The last event MUST end at the end of the video.
"""
            },
        ],
    }
]

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

generated_ids = model.generate(
    **inputs,
    max_new_tokens=2048,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(inputs["pixel_values_videos"].shape)

print(output_text[0])