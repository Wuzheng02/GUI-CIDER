import os
import json
import torch
import pynvml
import time
from tqdm import tqdm
from multiprocessing import Process, Queue, JoinableQueue
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= CONFIGURATION =================
MODEL_PATH = "/data4/Qwen3-VL-32B-Instruct"
BASE_DATA_DIR = "/data1/home/wuzheng/Trinity/data/Odyssey"
TARGET_FILES = [
    "subset_odyssey_9.json", "subset_odyssey_10.json", "subset_odyssey_11.json", "subset_odyssey_12.json", "subset_odyssey_13.json", "subset_odyssey_14.json", "subset_odyssey_15.json", "subset_odyssey_16.json",
    "subset_odyssey_17.json", "subset_odyssey_18.json"
]
# TARGET_FILES = [
#     "AC_test_easy.json", "AC_test_medium.json", "AC_test_hard.json",
#     "AITZ_test_easy.json", "AITZ_test_medium.json", "AITZ_test_hard.json",
#     "Odyssey_test_easy.json", "Odyssey_test_medium.json", "Odyssey_test_hard.json"
# ]

# BASE_DATA_DIR = "/data1/home/wuzheng/Trinity/data_preprocess"
# TARGET_FILES = [
#     "AITZ_train.json"
# ]
PROMPT_SEMANTIC_GROUNDING = """Role: You are an Expert UI Behavioral Analyst.
Task: Map a low-level execution token to a high-level semantic affordance.
Action: {action_raw}
the two coordinates of the click action are relative coordinates in thousandths: the first coordinate is the thousandths coordinate from left to right, and the second coordinate is the thousandths coordinate from top to bottom. For example, 500,500 means the exact center of the image.
Output: [Target Element]: (Name/Description) | [Action Intent]: (Detailed functional purpose)."""

PROMPT_WORLD_KNOWLEDGE = """Role: You are a Cognitive Science Researcher.
Task: Extract the latent world knowledge required for this step.
Task Goal: {task}
Semantic Action: {action_nl}

Output: A concise, academic synthesis of the required knowledge priors. Generate one to five pieces of knowledge as needed.
Strictly follow the format: each knowledge piece must be a separate line starting with "- ", with no additional commentary, numbering, or introductory text.
Output example:
- "Delete icon is typically a trash bin."
- "Dragging an item onto the trash bin triggers deletion."
"""

PROMPT_CAUSAL_LOGIC = """
Role: Expert GUI Causal Analyst & Data Architect.
Task: Generate a high-fidelity state transition trajectory to train a GUI Agent's causal reasoning.

Inputs:
- Task Goal: {task}
- Action (A_t): {action_nl}
- Visuals: Image 1 (Pre-state $S_{{t}}$), Image 2 (Post-state $S_{{t+1}}$).

Output Structure (Strictly follow this template):

[Pre-state Description]: 
(Brief one-sentence description of Image 1, focusing on the layout, the state of the target element, and relevant environmental context.)

[Post-state Description]: 
(Brief one-sentence description of Image 2, explicitly highlighting visual deltas such as new overlays, changed icons, or list updates compared to Image 1.)

[Action Effect]:
- Trigger: (What interaction is performed, e.g., "Clicking the 'Cart' icon at [x,y]")
- Mechanism: (What UI function is invoked, e.g., "Triggers a navigation event to the checkout page via the app's routing engine")

[Reasoning]: 
(A rigorous but brief explanation of why A_t was the optimal choice to fulfill the Task Goal given the state S_t, linking the affordance of the element to the objective.)
"""
# =================================================

class AcademicUIExtractorQwen3:
    def __init__(self, model_path, device_id):
        self.device_id = device_id
        print(f"[*] [GPU {device_id}] Loading Qwen3-VL from {model_path}...")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="bfloat16", 
            device_map={"": 0}, 
            attn_implementation="sdpa",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )

    def _inference(self, prompt, image_paths):
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_kwargs.get("video_metadata", None),
            padding=True,
            return_tensors="pt",
            do_resize=False 
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"[GPU {self.device_id}] Generated Output: {output_text}")
        return output_text.strip()

    def process_file(self, json_file):
        json_path = os.path.join(BASE_DATA_DIR, json_file)
        output_filename = f"extracted_{json_file.replace('.json', '.txt')}"
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"[GPU {self.device_id}] Processing {json_file} ({len(data)} steps)...")
        
        with open(output_filename, "w", encoding="utf-8") as f_out:
            for i in tqdm(range(len(data)), desc=f"GPU {self.device_id}"):
                item = data[i]
                img_path_rel = item['image_path']
                img_path_abs = os.path.normpath(os.path.join(BASE_DATA_DIR, img_path_rel.split("P-subset/")[-1]))
                
                if not os.path.exists(img_path_abs): continue

                # 1. Semantic Grounding
                action_nl = self._inference(PROMPT_SEMANTIC_GROUNDING.format(action_raw=item['action']), [img_path_abs])
                # 2. World Knowledge
                world_k = self._inference(PROMPT_WORLD_KNOWLEDGE.format(task=item['task'], action_nl=action_nl), [img_path_abs])
                # 3. Causal Analysis
                causal_k = "Final State: No subsequent transition."
                if i + 1 < len(data) and data[i+1]['task'] == item['task']:
                    next_img_path = os.path.normpath(os.path.join(BASE_DATA_DIR, data[i+1]['image_path'].split("P-subset/")[-1]))
                    if os.path.exists(next_img_path):
                        causal_k = self._inference(PROMPT_CAUSAL_LOGIC.format(task=item['task'], action_nl=action_nl), [img_path_abs, next_img_path])

                report = (
                    f"--- ENTRY: {json_file}_step_{i} ---\n"
                    f"[TASK]: {item['task']}\n"
                    f"[RAW ACTION]: {item['action']}\n"
                    f"[SEMANTIC GROUNDING]: {action_nl}\n"
                    f"[WORLD KNOWLEDGE]: {world_k}\n"
                    f"[CAUSAL DYNAMICS]: {causal_k}\n"
                    f"{'='*70}\n\n"
                )
                f_out.write(report)
                f_out.flush()

def worker(gpu_id, task_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        extractor = AcademicUIExtractorQwen3(MODEL_PATH, gpu_id)
        while True:
            json_file = task_queue.get()
            if json_file is None: # 结束信号
                break
            extractor.process_file(json_file)
            task_queue.task_done()
    except Exception as e:
        print(f"[Error] GPU {gpu_id} worker failed: {e}")
    finally:
        pynvml.nvmlShutdown()

def get_qualified_gpus(min_memory_gb=60):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    qualified = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = info.free / (1024**3)
        if free_gb > min_memory_gb:
            qualified.append(i)
    return qualified

if __name__ == "__main__":
    available_gpus = get_qualified_gpus(70)
    if not available_gpus:
        print("[!] No GPUs found with > 70GB free memory. Exiting.")
        exit()
    
    print(f"[*] Found {len(available_gpus)} qualified GPUs: {available_gpus}")

    task_queue = JoinableQueue()
    for f in TARGET_FILES:
        task_queue.put(f)

    processes = []
    for gpu_id in available_gpus:
        task_queue.put(None) 
        p = Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)
        time.sleep(10)

    for p in processes:
        p.join()

    print("[*] All academic extractions completed.")