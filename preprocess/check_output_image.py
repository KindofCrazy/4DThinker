import json
import sys

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "./data/merged_qa_all_converted.jsonl"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else INPUT_FILE.replace(".jsonl", "_cleaned.jsonl")

kept = []
removed = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        text_output = data.get("text_output", "")
        image_output = data.get("image_output", [])

        tag_count = text_output.count("<output_image>")
        img_count = len(image_output)

        if tag_count != img_count:
            removed.append(data)
            print(f"[Line {line_no}] MISMATCH: <output_image> count={tag_count}, image_output count={img_count}, id={data.get('id', 'N/A')}")
            print(f"  text_output (truncated): {text_output[:200]}...")
            print(f"  image_output: {image_output}")
            print()
        else:
            kept.append(line)

print(f"Total lines: {line_no}")
print(f"Kept: {len(kept)}")
print(f"Removed: {len(removed)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in kept:
        f.write(line + "\n")

print(f"\nCleaned file saved to: {OUTPUT_FILE}")
