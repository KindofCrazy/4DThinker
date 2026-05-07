import json
import sys


def convert_line(line: str) -> str:
    data = json.loads(line.strip())

    cot = data["cot"]
    cot = cot.replace("<thinking>", "<think>").replace("</thinking>", "</think>")

    new_data = {
        "id": data["id"],
        "text_input": data["question"],
        "text_output": cot,
        "image_input": data["original_image_paths"],
        "image_output": data["mask_image_paths"],
    }
    return json.dumps(new_data, ensure_ascii=False)


def convert_file(input_path: str, output_path: str):
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(convert_line(line) + "\n")
            count += 1
    print(f"Converted {count} lines: {input_path} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_format.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    convert_file(sys.argv[1], sys.argv[2])
