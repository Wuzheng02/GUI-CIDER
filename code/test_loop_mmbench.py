import os
import json
import torch
import re
import sys
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import defaultdict

# ================= CONFIGURATION =================
# MODEL_PATH = "/data1/home/wuzheng/Trinity/lora/qwen3vl_knowledge_aitz_v1"
# MODEL_PATH = "/data4/Qwen3-VL-32B-Instruct"
MODEL_PATH = "/data1/models/Qwen3-VL-8B-Instruct"
JSON_PATH = "/data1/home/wuzheng/Trinity/MMbench-GUI/L1_annotations.json"
IMAGE_ROOT = "/data1/home/wuzheng/Trinity/MMbench-GUI/offline_images"

PLATFORMS = ["os_android", "os_ios", "os_linux", "os_mac", "os_web", "os_windows"]
DIFFICULTIES = ["easy", "medium", "hard"]
# =================================================


# ================= LOG SETUP =================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = f"./eval_log_{timestamp}.txt"
log_file = open(LOG_PATH, "w", encoding="utf-8")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)
# ============================================


class MMBenchEvaluator:
    def __init__(self, model_path):
        print(f"[*] Loading Model from {model_path}...")
        print(f"[*] Log file: {LOG_PATH}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )

    def extract_answer(self, text):
        """从模型输出中提取选项字母"""
        match = re.search(r'[A-E]', text.upper())
        return match.group(0) if match else ""

    def run_eval(self):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        print(f"[*] Starting Evaluation on {len(data)} image items...")

        for item in tqdm(data):
            platform = item['platform']
            image_name = item['image_path']
            img_path = os.path.join(IMAGE_ROOT, platform, image_name)

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            for group in item['groups']:
                question = group['question']
                options_dict = group['options']
                gt_answer = group['answer'].strip().upper()
                difficulty = group['difficulty'].lower()

                options_str = ""
                for k, v in options_dict.items():
                    options_str += f"{k}. {v} "

                prompt = (
                    f"Based on the GUI screenshot, answer the following question:\n"
                    f"{question} Options: {options_str}\n"
                    f"Please output only the letter of the correct option (A, B, C, D, or E)."
                )

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt}
                    ]
                }]

                text_template = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.processor(
                    text=[text_template],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=16)
                    generated_ids_trimmed = [
                        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]

                # print(f"[DEBUG] Model Output: {output_text}")

                pred_letter = self.extract_answer(output_text)

                print(f"[DEBUG] GT Answer: {gt_answer}")
                print(f"[DEBUG] Predicted Answer: {pred_letter}")

                results[difficulty][platform]["total"] += 1
                print(f"[DEBUG] Total for {platform} ({difficulty}): {results[difficulty][platform]['total']}")

                if pred_letter == gt_answer:
                    results[difficulty][platform]["correct"] += 1

        self.print_report(results)

    def print_report(self, results):
        print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)

        for diff in DIFFICULTIES:
            line_parts = []
            diff_total_correct = 0
            diff_total_count = 0

            for plat in PLATFORMS:
                stats = results[diff][plat]
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
                line_parts.append(f"{plat}: {acc:.2f}%")

                diff_total_correct += stats['correct']
                diff_total_count += stats['total']

            overall_acc = (
                diff_total_correct / diff_total_count * 100
                if diff_total_count > 0 else 0.0
            )

            print(f"{diff.upper()}: {', '.join(line_parts)} | Overall: {overall_acc:.2f}%")


if __name__ == "__main__":
    print("="*60)
    print("[RUN CONFIG]")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"LOG_PATH: {LOG_PATH}")
    print("="*60)

    evaluator = MMBenchEvaluator(MODEL_PATH)
    evaluator.run_eval()

    log_file.close()
