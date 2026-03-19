# Inference (tối giản)

Tài liệu này mô tả cách nạp base model + LoRA adapter và tạo đầu ra cơ bản.

## Chuẩn bị
- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Adapter: `artifacts/training/` (output sau khi train)

## Ví dụ code
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_DIR = "artifacts/training"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

system_prompt = (
    "You are a Python bug detection expert. "
    "When given a piece of Python code, you must identify the bug type, "
    "provide a high-level description of the issue, and supply the corrected code."
)

user_code = \"\"\"def add(a,b):\n    return a-b\n\"\"\"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f\"Analyze this code:\\n```python\\n{user_code}\\n```\"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Lưu ý
- Output cần review thủ công, không dùng trực tiếp trong production.
- Nếu GPU không hỗ trợ `bfloat16`, bạn có thể đổi `torch_dtype` sang `torch.float16`.
