from PIL import Image
import torch
import openvino as ov
import time
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils.chat_template_utils import render_jinja_template
from ov_paddleocr_vl import PaddleOCR_VL_OV, OVPaddleOCRVLForCausalLM
from image_processing_paddleocr_vl import PaddleOCRVLImageProcessor

# paddleocr_vl_ov = PaddleOCR_VL_OV(pretrained_model_path="./PaddleOCR-VL", ov_model_path="./ov_paddleocr_vl_model", device="cpu", llm_int4_compress=True, vision_int8_quant=False)
# paddleocr_vl_ov.export_vision_to_ov()

# ---- Settings ----
model_path = "./PaddleOCR-VL"
image_path = "./paddle_ocr_vl.png"
image_path = "./paddle_arch.jpg"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula'
# ------------------

DEVICE = "cpu"

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

chat_template = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not eos_token is defined -%}\n    {%- set eos_token = "</s>" -%}\n{%- endif -%}\n{%- if not image_token is defined -%}\n    {%- set image_token = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " -}}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "image" -%}\n                {{ image_token }}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] }}\n            {%- endif -%}\n        {%- endfor -%}\n        {{ "\\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " -}}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] }}\n            {%- endif -%}\n        {%- endfor -%}\n        {{ eos_token -}}\n    {%- elif message["role"] == "system" -%}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] + "\\n" }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}\n'

image = Image.open(image_path).convert("RGB")
image = image.resize((1200, 800), Image.Resampling.LANCZOS)

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user",         
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]


inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)


# 统计生成和解码的执行时间
start_time = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
generate_time = time.perf_counter() - start_time

start_time = time.perf_counter()
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
decode_time = time.perf_counter() - start_time

total_time = generate_time + decode_time

print("\n" + "="*60)
print("📄 Transformers OCR 识别结果:")
print("="*60)
print(outputs)
print("="*60)
print(f"\n⏱️  执行时间统计:")
print(f"   - 生成时间 (generate): {generate_time:.3f} 秒 ({generate_time*1000:.2f} 毫秒)")
print(f"   - 解码时间 (decode):   {decode_time:.3f} 秒 ({decode_time*1000:.2f} 毫秒)")
print(f"   - 总时间:              {total_time:.3f} 秒 ({total_time*1000:.2f} 毫秒)")
print("="*60 + "\n")

# breakpoint()

llm_infer_list = []
vision_infer = []
core = ov.Core()
paddleocr_vl_model = OVPaddleOCRVLForCausalLM(core=core, ov_model_path="./ov_paddleocr_vl_model", device="GPU", llm_int4_compress=False, vision_int8_quant=False, llm_int8_quant=False, llm_infer_list=llm_infer_list, vision_infer=vision_infer)
text, generation_indices = render_jinja_template(
    conversations=[messages],
    chat_template=chat_template,
    add_generation_prompt=True, 	
    return_tensors="pt",
)
my_preprocer = PaddleOCRVLImageProcessor(resample=3, rescale_factor=0.00392156862745098, image_mean=[0.5,0.5,0.5],image_std=[0.5,0.5,0.5], min_pixels=147384, \
    max_pixels=2822400, patch_size=14, temporal_patch_size=1, merge_size=2)
images_info = my_preprocer(images=image, return_tensors="pt")

if not isinstance(text, list):
    text = [text]
index = 0
for i in range(len(text)):
    while "<|IMAGE_PLACEHOLDER|>" in text[i]:
        text[i] = text[i].replace(
            "<|IMAGE_PLACEHOLDER|>",
            "<|placeholder|>"
            * (
                images_info['image_grid_thw'][index].prod()
                // 2
                // 2
            ),
            1,
        )
        index += 1
    text[i] = text[i].replace("<|placeholder|>", "<|IMAGE_PLACEHOLDER|>")

text_inputs = paddleocr_vl_model.tokenizer(text, return_tensors="pt")

version = ov.get_version()
print("OpenVINO version \n", version)
print('\n')

generation_config = {
    "bos_token_id": paddleocr_vl_model.tokenizer.bos_token_id,
    "eos_token_id": paddleocr_vl_model.tokenizer.eos_token_id,
    "pad_token_id": paddleocr_vl_model.tokenizer.pad_token_id,
    "max_new_tokens": 1024,
    "do_sample": False,
}
# 统计 chat 方法的执行时间
start_time = time.perf_counter()
response, history = paddleocr_vl_model.chat(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"], pixel_values=images_info["pixel_values"], image_grid_thw=images_info["image_grid_thw"], generation_config=generation_config)
chat_time = time.perf_counter() - start_time

print("\n" + "="*60)
print("📄 openVINO OCR 识别结果:")
print("="*60)
print(response)
print("="*60)
print(f"\n⏱️  执行时间统计:")
print(f"   - Chat 方法执行时间: {chat_time:.3f} 秒 ({chat_time*1000:.2f} 毫秒)")
print("="*60 + "\n")