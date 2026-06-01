from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    import torch
    from transformers import AutoProcessor
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install at least: pip install -U transformers accelerate pillow torch"
    ) from exc


DEFAULT_MODEL_PATH = "/data1/models/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_ROOT = "/data1/home/wuzheng/Trinity/GUI_knowledge_bench/Image"
DEFAULT_JSON_ROOT = "/data1/home/wuzheng/Trinity/GUI_knowledge_bench/KnowledgeBench"
DEFAULT_OUTPUT_DIR = "qwen3_vl_8b_gui_kb_results"

ORDERED_DIMENSIONS = [
    "InterfacePerception/StateInformationUnderstanding",
    "InterfacePerception/WidgetFunctionUnderstanding",
    "InterfacePerception/LayoutSemanticsUnderstanding",
    "InteractionPrediction/ActionEffect",
    "InteractionPrediction/ActionPrediction",
    "InteractionPrediction/ActionPrediction-Parameter",
    "InstructionUnderstanding/GoalInterpretation",
    "InstructionUnderstanding/TaskPlanning",
    "Overall",
]

PAPER_LABELS = {
    "InterfacePerception/StateInformationUnderstanding": ("Interface Knowledge", "state"),
    "InterfacePerception/WidgetFunctionUnderstanding": ("Interface Knowledge", "widget"),
    "InterfacePerception/LayoutSemanticsUnderstanding": ("Interface Knowledge", "layout"),
    "InteractionPrediction/ActionEffect": ("Interaction Knowledge", "effect"),
    "InteractionPrediction/ActionPrediction": ("Interaction Knowledge", "type"),
    "InteractionPrediction/ActionPrediction-Parameter": ("Interaction Knowledge", "parameter"),
    "InstructionUnderstanding/GoalInterpretation": ("Procedure Knowledge", "objective"),
    "InstructionUnderstanding/TaskPlanning": ("Procedure Knowledge", "workflow"),
    "Overall": ("Overall", "overall"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B-Instruct on GUI Knowledge Bench with local transformers inference."
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--json-root", default=DEFAULT_JSON_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: evaluate first N questions.")
    parser.add_argument("--resume", action="store_true", help="Skip questions whose result JSON already exists.")
    parser.add_argument("--no-visual-prompt", action="store_true", help="Do not draw annotation boxes.")
    parser.add_argument("--knowledge-prompt", action="store_true", help="Append needed_knowledge tips when present.")
    parser.add_argument("--recursive-image-search", action="store_true", help="Build a basename image index under image-root.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_processor(args: argparse.Namespace):
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )

    load_kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }

    last_error = None
    candidates = []
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        try:
            module = __import__("transformers", fromlist=[class_name])
            candidates.append(getattr(module, class_name))
        except Exception:
            continue

    for model_cls in candidates:
        try:
            model = model_cls.from_pretrained(str(model_path), **load_kwargs)
            model.eval()
            return model, processor
        except Exception as exc:  # try next compatible auto class
            last_error = exc

    raise RuntimeError(f"Could not load model from {model_path}. Last error: {last_error}")


def load_questions(json_root: Path) -> list[dict[str, Any]]:
    questions = []
    for path in sorted(json_root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] Skip invalid JSON {path}: {exc}", file=sys.stderr)
            continue
        data["_json_path"] = str(path)
        questions.append(data)
    return questions


def build_image_index(image_root: Path) -> dict[str, Path]:
    index = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            index.setdefault(path.name, path)
    return index


def resolve_image(path_text: str, image_root: Path, image_index: dict[str, Path] | None = None) -> Path:
    raw = Path(path_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(image_root / path_text)
    candidates.append(image_root / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if image_index and raw.name in image_index:
        return image_index[raw.name].resolve()

    raise FileNotFoundError(f"Image not found: {path_text}")


def draw_annotation_if_needed(question: dict[str, Any], image_path: Path, output_dir: Path) -> Path:
    annotation = question.get("annotation") or {}
    bbox = annotation.get("bbox")
    if not bbox or len(bbox) != 4:
        return image_path

    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotated_dir / f"{question.get('question_id', image_path.stem)}.png"
    if out_path.exists():
        return out_path

    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(4, round(min(img.size) * 0.006))
        for offset in range(width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(255, 0, 0))
        img.save(out_path)
    return out_path


def image_content(path: Path) -> dict[str, str]:
    return {"type": "image", "image": str(path)}


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def options_for_question(question: dict[str, Any]) -> list[dict[str, Any]]:
    q_type = question.get("question_type")
    if q_type == "yes_or_no":
        return [{"text": "yes"}, {"text": "no"}, {"text": "unknown"}]

    texts = question.get("option_text") or []
    images = question.get("option_image_dir_list") or []
    count = max(len(texts), len(images))
    options = []
    for idx in range(count):
        options.append(
            {
                "text": texts[idx] if idx < len(texts) else "",
                "image": images[idx] if idx < len(images) else "",
            }
        )
    return options


def answer_schema(question: dict[str, Any], options: list[dict[str, Any]]) -> str:
    if question.get("question_type") == "yes_or_no":
        return '{"answer": "<yes/no/unknown>"}'
    labels = "/".join(option_label(i) for i in range(len(options)))
    return '{"answer": "<' + labels + '>"}'


def system_prompt(question: dict[str, Any]) -> str:
    knowledge = question.get("knowledge", {})
    ktype = knowledge.get("knowledge_type")
    subtype = knowledge.get("knowledge_sub_type")

    if ktype == "InstructionUnderstanding" and subtype == "GoalInterpretation":
        return (
            "You are a Graphical User Interface (GUI) agent. You will be given a sequence "
            "of screenshots and a task instruction. Select whether the task is completed: "
            "yes, no, or unknown."
        )
    if ktype == "InstructionUnderstanding" and subtype == "TaskPlanning":
        return (
            "You are a Graphical User Interface (GUI) agent. You will be given a task, "
            "screenshots, and candidate GUI operations. Select the best answer."
        )
    if ktype == "InteractionPrediction":
        return (
            "You are a Graphical User Interface (GUI) agent. You will reason about GUI "
            "actions, screenshots, and possible action effects. Select the correct option."
        )
    return (
        "You are a Graphical User Interface (GUI) agent. You will be given screenshot(s), "
        "a question, and options. Select the correct answer."
    )


def task_text(question: dict[str, Any]) -> str:
    knowledge = question.get("knowledge", {})
    subtype = knowledge.get("knowledge_sub_type")
    q_text = question.get("question_text", "")

    if subtype == "GoalInterpretation":
        return f'According to the screenshots, has the task "{q_text}" been completed?'
    if subtype == "ActionEffect":
        action = q_text.replace("ActionEffect: ", "").replace("ActionEffect:", "")
        return f"After performing the described action `{action}`, which option is the resulting screenshot?"
    if subtype == "ActionPrediction":
        if "ActionPrediction-Parameter:" in q_text:
            action = q_text.replace("ActionPrediction-Parameter: ", "")
            return (
                "The screenshots show a transition. Select the option containing the correct "
                f"parameter for action `{action}`."
            )
        if "ActionPrediction-Type:" in q_text:
            return "The screenshots show a transition. Select which action was performed."
    return q_text


def paper_dimension_for_question(question: dict[str, Any]) -> str:
    knowledge = question.get("knowledge", {})
    ktype = knowledge.get("knowledge_type", "Unknown")
    subtype = knowledge.get("knowledge_sub_type", "Unknown")
    q_text = question.get("question_text", "")
    if ktype == "InteractionPrediction" and subtype == "ActionPrediction":
        if "ActionPrediction-Parameter:" in q_text:
            return "InteractionPrediction/ActionPrediction-Parameter"
        return "InteractionPrediction/ActionPrediction"
    return f"{ktype}/{subtype}"


def build_messages(
    question: dict[str, Any],
    image_root: Path,
    image_index: dict[str, Path] | None,
    output_dir: Path,
    use_visual_prompt: bool,
    use_knowledge_prompt: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    content = []
    image_paths_for_debug = []
    q_images = question.get("question_image_dir_list") or []
    if isinstance(q_images, str):
        q_images = [q_images]

    for idx, image_name in enumerate(q_images):
        resolved = resolve_image(image_name, image_root, image_index)
        if idx == 0 and use_visual_prompt:
            resolved = draw_annotation_if_needed(question, resolved, output_dir)
        content.append(image_content(resolved))
        image_paths_for_debug.append(str(resolved))

    content.append(text_content(task_text(question).strip() + "\n"))

    options = options_for_question(question)
    if question.get("question_type") == "multiple_choice":
        for idx, opt in enumerate(options):
            label = option_label(idx)
            text = opt.get("text") or ""
            content.append(text_content(f"{label}. {text}\n"))
            if opt.get("image"):
                opt_img = resolve_image(opt["image"], image_root, image_index)
                content.append(image_content(opt_img))
                image_paths_for_debug.append(str(opt_img))
    else:
        content.append(text_content("Options: yes, no, unknown.\n"))

    final_instruction = (
        "Answer strictly with JSON only. Do not output extra text. "
        f"Schema: {answer_schema(question, options)}"
    )
    if use_knowledge_prompt and question.get("needed_knowledge"):
        final_instruction += f"\nHelpful GUI knowledge: {question['needed_knowledge']}"
    content.append(text_content(final_instruction))

    messages = [
        {"role": "system", "content": [text_content(system_prompt(question))]},
        {"role": "user", "content": content},
    ]
    return messages, image_paths_for_debug


def move_inputs_to_device(inputs: Any, model: Any) -> Any:
    if not hasattr(inputs, "to"):
        return inputs
    try:
        return inputs.to(model.device)
    except Exception:
        return inputs


def prepare_inputs(processor: Any, messages: list[dict[str, Any]], model: Any) -> Any:
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return move_inputs_to_device(inputs, model)
    except Exception:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Processor could not directly tokenize multimodal messages, and qwen-vl-utils "
                "is not installed. Install with: pip install qwen-vl-utils"
            ) from exc

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs = {
            "text": [text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            kwargs["videos"] = video_inputs
        inputs = processor(**kwargs)
        return move_inputs_to_device(inputs, model)


def decode_generated(processor: Any, inputs: Any, generated_ids: Any) -> str:
    try:
        input_len = inputs.input_ids.shape[-1]
        generated_ids = generated_ids[:, input_len:]
    except Exception:
        pass
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def infer_answer(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = prepare_inputs(processor, messages, model)
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    return decode_generated(processor, inputs, generated_ids)


def extract_answer(raw_text: str, question: dict[str, Any]) -> str:
    text = raw_text.strip()
    try:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("answer") is not None:
                return normalize_answer(obj["answer"], question)
    except Exception:
        pass

    if question.get("question_type") == "yes_or_no":
        match = re.search(r"\b(yes|no|unknown)\b", text, flags=re.IGNORECASE)
        return normalize_answer(match.group(1), question) if match else ""

    labels = [option_label(i) for i in range(len(options_for_question(question)))]
    match = re.search(r"\b(" + "|".join(labels) + r")\b", text, flags=re.IGNORECASE)
    return normalize_answer(match.group(1), question) if match else ""


def normalize_answer(value: Any, question: dict[str, Any] | None = None) -> str:
    answer = str(value).strip().strip('"').strip("'").lower()
    if answer in {"true", "y"}:
        answer = "yes"
    if answer in {"false", "n"}:
        answer = "no"
    if question and question.get("question_type") == "multiple_choice":
        answer = answer.upper()
    return answer


def score_item(question: dict[str, Any], prediction: str, success: bool) -> bool:
    gt = normalize_answer(question.get("groundtruth", ""), question)
    return bool(success and prediction and prediction == gt)


def result_path_for(output_dir: Path, question: dict[str, Any]) -> Path:
    qid = question.get("question_id") or Path(question.get("_json_path", "unknown")).stem
    return output_dir / "predictions" / f"{qid}.json"


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    json_root = Path(args.json_root).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    questions = load_questions(json_root)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"[info] Loaded {len(questions)} questions from {json_root}")

    image_index = build_image_index(image_root) if args.recursive_image_search else None
    if image_index is not None:
        print(f"[info] Indexed {len(image_index)} images under {image_root}")

    model, processor = load_model_and_processor(args)
    rows = []
    errors = []

    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id") or Path(question.get("_json_path", "")).stem
        out_path = result_path_for(output_dir, question)
        if args.resume and out_path.exists():
            try:
                result = json.loads(out_path.read_text(encoding="utf-8"))
                rows.append(result)
                print(f"[{idx}/{len(questions)}] skip existing {qid}")
                continue
            except Exception:
                pass

        start = time.time()
        try:
            messages, image_debug = build_messages(
                question=question,
                image_root=image_root,
                image_index=image_index,
                output_dir=output_dir,
                use_visual_prompt=not args.no_visual_prompt,
                use_knowledge_prompt=args.knowledge_prompt,
            )
            raw = infer_answer(
                model=model,
                processor=processor,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred = extract_answer(raw, question)
            success = bool(pred)
            error = None
        except Exception:
            raw = ""
            pred = ""
            success = False
            image_debug = []
            error = traceback.format_exc()
            errors.append({"id": qid, "error": error})

        elapsed = round(time.time() - start, 3)
        gt = normalize_answer(question.get("groundtruth", ""), question)
        correct = score_item(question, pred, success)
        knowledge = question.get("knowledge", {})
        paper_dimension = paper_dimension_for_question(question)
        paper_category, paper_metric = PAPER_LABELS.get(paper_dimension, (knowledge.get("knowledge_type", "Unknown"), paper_dimension))
        result = {
            "id": qid,
            "json_path": question.get("_json_path"),
            "knowledge_type": knowledge.get("knowledge_type", "Unknown"),
            "knowledge_sub_type": knowledge.get("knowledge_sub_type", "Unknown"),
            "paper_category": paper_category,
            "paper_metric": paper_metric,
            "paper_dimension": paper_dimension,
            "question_text": question.get("question_text", ""),
            "question_type": question.get("question_type"),
            "groundtruth": gt,
            "answer": pred,
            "is_success": success,
            "is_correct": correct,
            "raw_response": raw,
            "images": image_debug,
            "elapsed_seconds": elapsed,
            "error": error,
        }
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(result)

        status = "OK" if correct else "WRONG"
        if not success:
            status = "FAIL"
        print(f"[{idx}/{len(questions)}] {status} {qid} pred={pred!r} gt={gt!r} time={elapsed}s")

    if errors:
        (output_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in errors) + "\n",
            encoding="utf-8",
        )
    return questions, rows


def summarize(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    buckets = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    for row in rows:
        dimension = row.get("paper_dimension")
        if not dimension:
            ktype = row.get("knowledge_type", "Unknown")
            subtype = row.get("knowledge_sub_type", "Unknown")
            dimension = f"{ktype}/{subtype}"
        for key in ("Overall", dimension):
            buckets[key]["total"] += 1
            buckets[key]["correct"] += int(bool(row.get("is_correct")))
            buckets[key]["failed"] += int(not bool(row.get("is_success")))

    summary = {}
    for key, stats in buckets.items():
        total = stats["total"]
        summary[key] = {
            "total": total,
            "correct": stats["correct"],
            "accuracy": stats["correct"] / total if total else 0.0,
            "failed": stats["failed"],
        }

    ordered_keys = [key for key in ORDERED_DIMENSIONS if key in summary]
    ordered_keys += sorted(key for key in summary if key not in ordered_keys)

    report = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "json_root": str(Path(args.json_root).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()),
        "output_dir": str(output_dir),
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "visual_prompt": not args.no_visual_prompt,
            "knowledge_prompt": args.knowledge_prompt,
        },
        "summary": {key: summary[key] for key in ordered_keys},
        "items": rows,
    }

    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Qwen3-VL-8B GUI Knowledge Bench Report",
        "",
        f"- Model: `{report['model_path']}`",
        f"- JSON root: `{report['json_root']}`",
        f"- Image root: `{report['image_root']}`",
        "",
        "| Knowledge | Paper Item | Internal Dimension | Total | Correct | Accuracy | Failed |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key in ordered_keys:
        stats = summary[key]
        paper_category, paper_metric = PAPER_LABELS.get(key, ("Other", key))
        md_lines.append(
            f"| {paper_category} | {paper_metric} | {key} | {stats['total']} | {stats['correct']} | {stats['accuracy']:.4f} | {stats['failed']} |"
        )
    (output_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with (output_dir / "per_item.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "knowledge_type",
                "knowledge_sub_type",
                "paper_category",
                "paper_metric",
                "paper_dimension",
                "question_type",
                "groundtruth",
                "answer",
                "is_success",
                "is_correct",
                "elapsed_seconds",
                "json_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(f"[done] Wrote {output_dir / 'report.md'}")
    print(f"[done] Wrote {output_dir / 'report.json'}")
    print(f"[done] Wrote {output_dir / 'per_item.csv'}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _, rows = evaluate(args)
    summarize(rows, output_dir, args)


if __name__ == "__main__":
    main()
