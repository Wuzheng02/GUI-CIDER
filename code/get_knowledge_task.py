import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================== 配置区 ==================
FILE_PATHS = [
    "/data1/home/wuzheng/Trinity/data_preprocess/AC_train.json",
    "/data1/home/wuzheng/Trinity/data_preprocess/AITZ_train.json",
    "/data1/home/wuzheng/Trinity/data_preprocess/Odyssey_train.json",
]
OUTPUT_TASK_TXT = "task.txt"
OUTPUT_TASK_JSON = "task.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_WORKERS = 20
BATCH_SIZE = 10
# ===========================================

def load_unique_tasks(file_paths):
    tasks = []
    seen = set()
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "task" in item:
                    task = item["task"]
                    if task not in seen:
                        seen.add(task)
                        tasks.append(task)
    return tasks

def save_task_txt(tasks, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"task.txt saved with {len(tasks)} tasks.")

def generate_step_list(client, task):
    prompt = f"""You are an expert in GUI task planning. Given a task description, break it down into a detailed step list. Each step should be a clear atomic action described in English. Output ONLY a JSON object in the following format:
{{"step_list": ["step 1", "step 2", ...]}}

Task description: {task}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in GUI operation planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            step_list = result.get("step_list", [])
            if isinstance(step_list, list):
                return task, step_list
            else:
                raise ValueError("step_list is not a list")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to process task after {MAX_RETRIES} attempts: {task}, error: {e}")
                return task, []

def load_existing_results(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["task"]: item["step_list"] for item in data}
    except Exception:
        print(f"Warning: Failed to load {filepath}, starting from scratch.")
        return {}

def save_results(results_dict, filepath):
    result_list = [{"task": t, "step_list": s} for t, s in results_dict.items()]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

def main():
    all_tasks = load_unique_tasks(FILE_PATHS)
    print(f"Total unique tasks: {len(all_tasks)}")
    save_task_txt(all_tasks, OUTPUT_TASK_TXT)

    completed = load_existing_results(OUTPUT_TASK_JSON)
    remaining_tasks = [t for t in all_tasks if t not in completed]
    print(f"Already completed: {len(completed)}, remaining: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("All tasks are already completed.")
        return

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    pbar = tqdm(total=len(remaining_tasks), desc="Processing tasks")
    pending = set(remaining_tasks) 

    batch_counter = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(generate_step_list, client, task): task
            for task in remaining_tasks
        }

        for future in as_completed(future_to_task):
            task_desc = future_to_task[future]
            try:
                task, step_list = future.result()
                completed[task] = step_list
                pbar.update(1)
                batch_counter += 1

          
                if batch_counter >= BATCH_SIZE or len(completed) == len(all_tasks):
                    save_results(completed, OUTPUT_TASK_JSON)
                    batch_counter = 0
            except Exception as e:
                print(f"Unexpected error for task: {task_desc}, {e}")
                completed[task_desc] = [] 
                pbar.update(1)
                batch_counter += 1
                if batch_counter >= BATCH_SIZE or len(completed) == len(all_tasks):
                    save_results(completed, OUTPUT_TASK_JSON)
                    batch_counter = 0

    pbar.close()
    save_results(completed, OUTPUT_TASK_JSON)
    print(f"All done. Final results saved to {OUTPUT_TASK_JSON}")

if __name__ == "__main__":
    main()
