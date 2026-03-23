import argparse
import json
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import dotenv
import langchain_core.exceptions
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

if os.path.exists('.env'):
    dotenv.load_dotenv()


STRICT_FILTER_SYSTEM_PROMPT = """You are a strict paper relevance classifier.

Task:
Classify whether a paper is directly relevant to ONE of the following core themes:
1) Large-model decision making / planning / policy optimization
2) Large-model reasoning (explicit reasoning capability improvement or evaluation)
3) Large-model post-training (e.g., SFT, RLHF, DPO, GRPO, alignment/post-training recipes)

Strict rules:
- Keep only if the paper's MAIN contribution is clearly about one of the 3 themes above.
- If relevance is weak, indirect, or unclear, set keep=false.
- If the paper just uses LLMs as a tool but does not focus on these themes as the core contribution, set keep=false.
- If uncertain, prefer keep=false.
"""


STRICT_FILTER_HUMAN_PROMPT = """Classify the following paper.

Title: {title}
Categories: {categories}
Abstract: {summary}

Return structured output only in valid json format.
Keep reason concise (<= 40 words).
"""


class TopicDecision(BaseModel):
    # Defaults make parser robust when provider omits optional fields.
    keep: bool = Field(default=False, description="Whether to keep this paper")
    confidence: int = Field(default=50, description="Confidence score from 0 to 100")
    theme: str = Field(default="none", description="One of: decision_making, reasoning, post_training, none")
    reason: str = Field(default="", description="Short reason for the decision")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        try:
            score = int(value)
        except Exception:
            score = 50
        return max(0, min(100, score))

    @field_validator("theme", mode="before")
    @classmethod
    def normalize_theme(cls, value):
        allowed = {"decision_making", "reasoning", "post_training", "none"}
        normalized = str(value or "none").strip().lower().replace(" ", "_")
        return normalized if normalized in allowed else "none"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=str, default="", help="Output JSONL path")
    parser.add_argument("--report", type=str, default="", help="Report JSON path")
    parser.add_argument("--max_workers", type=int, default=6, help="Parallel workers")
    return parser.parse_args()


def get_structured_output_method() -> str:
    configured = os.environ.get("STRUCTURED_OUTPUT_METHOD", "").strip()
    if configured:
        return configured

    base_url = os.environ.get("OPENAI_BASE_URL", "").lower()
    if "dashscope.aliyuncs.com" in base_url:
        return "json_mode"

    return "function_calling"


def build_chain(model_name: str):
    method = get_structured_output_method()
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=1,
        timeout=45,
    ).with_structured_output(
        TopicDecision,
        method=method,
    )
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(STRICT_FILTER_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(STRICT_FILTER_HUMAN_PROMPT),
    ])
    chain = prompt_template | llm
    return chain, method


def classify_item(chain, item: Dict) -> Tuple[bool, Dict]:
    title = str(item.get("title", "")).strip()
    categories = item.get("categories", [])
    categories_text = ", ".join(categories) if isinstance(categories, list) else str(categories)
    summary = str(item.get("summary", "")).strip()

    # Keep prompt size bounded to reduce latency/cost and avoid provider limits.
    if len(summary) > 2500:
        summary = summary[:2500]

    payload = {
        "title": title,
        "categories": categories_text,
        "summary": summary,
    }

    try:
        response: TopicDecision = chain.invoke(payload)
        decision = {
            "keep": bool(response.keep),
            "confidence": int(response.confidence),
            "theme": str(response.theme),
            "reason": str(response.reason),
        }
        return decision["keep"], decision
    except langchain_core.exceptions.OutputParserException as e:
        # Fallback: try parsing the raw JSON that some providers return
        # when optional fields are omitted.
        error_msg = str(e)
        match = re.search(r"from completion\\s*(\\{.*\\})\\.\\s*Got:", error_msg, re.DOTALL)
        if not match:
            raise

        parsed = json.loads(match.group(1))
        decision = {
            "keep": bool(parsed.get("keep", False)),
            "confidence": int(parsed.get("confidence", 50) or 50),
            "theme": str(parsed.get("theme", "none") or "none"),
            "reason": str(parsed.get("reason", "") or ""),
        }
        decision["confidence"] = max(0, min(100, decision["confidence"]))
        if decision["theme"] not in {"decision_making", "reasoning", "post_training", "none"}:
            decision["theme"] = "none"
        return decision["keep"], decision


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"Input file does not exist: {args.data}", file=sys.stderr)
        sys.exit(2)

    output_path = args.output or args.data.replace(".jsonl", "_topic_filtered.jsonl")
    report_path = args.report or args.data.replace(".jsonl", "_topic_filter_report.json")

    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    chain, method = build_chain(model_name)
    print(f"Topic filter model: {model_name}, structured_output_method={method}", file=sys.stderr)

    items: List[Dict] = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if not items:
        print("No items found in input file.", file=sys.stderr)
        # write empty output/report for consistency
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_file": args.data,
                    "output_file": output_path,
                    "model": model_name,
                    "structured_output_method": method,
                    "total": 0,
                    "kept": 0,
                    "dropped": 0,
                    "errors": 0,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        sys.exit(1)

    kept_items: List[Dict] = []
    error_count = 0
    drop_count = 0
    decisions: Dict[str, Dict] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_idx = {
            executor.submit(classify_item, chain, item): idx
            for idx, item in enumerate(items)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            item = items[idx]
            paper_id = item.get("id", f"idx_{idx}")

            try:
                keep, decision = future.result()
                decisions[paper_id] = decision
                if keep:
                    kept_items.append(item)
                else:
                    drop_count += 1
            except Exception as e:
                # Strict mode: classification failure is treated as drop.
                error_count += 1
                drop_count += 1
                decisions[paper_id] = {
                    "keep": False,
                    "confidence": 0,
                    "theme": "none",
                    "reason": f"classification_error: {e}",
                }
                print(f"Topic filter error for {paper_id}: {e}", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in kept_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input_file": args.data,
        "output_file": output_path,
        "model": model_name,
        "structured_output_method": method,
        "total": len(items),
        "kept": len(kept_items),
        "dropped": drop_count,
        "errors": error_count,
        "decisions": decisions,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"Topic filter finished: total={len(items)}, kept={len(kept_items)}, dropped={drop_count}, errors={error_count}",
        file=sys.stderr,
    )

    if len(items) > 0 and error_count == len(items):
        print("All topic classifications failed.", file=sys.stderr)
        sys.exit(2)

    if len(kept_items) == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
