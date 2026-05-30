#!/usr/bin/env python
import ast
from pathlib import Path


DSR_EVAL = Path(__file__).resolve().parents[1] / "dsr_eval.py"


def load_frame_transform():
    tree = ast.parse(DSR_EVAL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply_video_mode_to_frames":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"List": list}
            exec(compile(module, str(DSR_EVAL), "exec"), namespace)
            return namespace[node.name]
    raise AssertionError("apply_video_mode_to_frames is not defined")


def main():
    transform = load_frame_transform()
    frames = ["f0.jpg", "f1.jpg", "f2.jpg"]
    assert transform(frames, "normal") == frames
    assert transform(frames, "repeat_first") == ["f0.jpg", "f0.jpg", "f0.jpg"]
    assert transform(frames, "reversed_video") == ["f2.jpg", "f1.jpg", "f0.jpg"]
    assert frames == ["f0.jpg", "f1.jpg", "f2.jpg"]


if __name__ == "__main__":
    main()
