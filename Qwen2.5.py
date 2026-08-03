from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
                """
You are an autonomous driving analyst.

Your task is to produce dense video captions for the ego vehicle.

Output format:

[0.0 - 1.5 s]
Motion:
Explanation:

[1.5 - 3.2 s]
Motion:
Explanation:

[3.2 - 5.7 s]
Motion:
Explanation:

Rules:
- Cover the entire video.
- Divide the video whenever the ego vehicle changes motion.
- Ignore other vehicles.
- Focus only on the ego vehicle.
- Motions include:
  - straight
  - slight left steering
  - slight right steering
  - left turn
  - right turn
  - acceleration
  - deceleration
  - stop
  - lane change
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
    max_new_tokens=512,
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

print(output_text[0])