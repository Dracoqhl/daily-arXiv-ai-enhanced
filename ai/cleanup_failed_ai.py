import argparse
import glob
import json
import os
import sys
from typing import Dict, List

FAILED_TLDR_VALUES = {
    "Summary generation failed",
    "Processing failed",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing jsonl files")
    parser.add_argument(
        "--file_list",
        type=str,
        default="assets/file-list.txt",
        help="Path to file-list.txt to regenerate",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional path to write cleanup report JSON",
    )
    return parser.parse_args()


def is_failed_item(item: Dict) -> bool:
    ai_data = item.get("AI")
    if not isinstance(ai_data, dict):
        return True

    tldr = str(ai_data.get("tldr", "")).strip()
    if tldr in FAILED_TLDR_VALUES:
        return True

    return False


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # Corrupted lines are treated as invalid and removed.
                continue
    return records


def write_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def regenerate_file_list(data_dir: str, file_list_path: str):
    os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    with open(file_list_path, "w", encoding="utf-8") as f:
        for path in jsonl_files:
            f.write(f"{os.path.basename(path)}\n")


def main():
    args = parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(0)

    target_files = sorted(glob.glob(os.path.join(data_dir, "*_AI_enhanced_*.jsonl")))

    report = {
        "data_dir": data_dir,
        "checked_files": len(target_files),
        "changed_files": 0,
        "deleted_files": 0,
        "removed_records": 0,
        "kept_records": 0,
        "details": {},
    }

    for file_path in target_files:
        records = load_jsonl(file_path)
        before = len(records)
        kept = [item for item in records if not is_failed_item(item)]
        after = len(kept)
        removed = before - after

        if removed > 0:
            report["changed_files"] += 1
            report["removed_records"] += removed

            if after == 0:
                os.remove(file_path)
                report["deleted_files"] += 1
                report["details"][os.path.basename(file_path)] = {
                    "before": before,
                    "after": 0,
                    "removed": removed,
                    "action": "deleted_empty_file",
                }
            else:
                write_jsonl(file_path, kept)
                report["details"][os.path.basename(file_path)] = {
                    "before": before,
                    "after": after,
                    "removed": removed,
                    "action": "rewritten",
                }
        else:
            report["details"][os.path.basename(file_path)] = {
                "before": before,
                "after": after,
                "removed": 0,
                "action": "unchanged",
            }

        report["kept_records"] += after

    regenerate_file_list(data_dir, args.file_list)

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"Cleanup finished: checked_files={report['checked_files']}, changed_files={report['changed_files']}, "
        f"removed_records={report['removed_records']}, deleted_files={report['deleted_files']}",
        file=sys.stderr,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
