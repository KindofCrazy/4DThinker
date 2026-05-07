import os
from PIL import Image

MAX_PIXELS = 256 * 28 * 28
VIDEO_MAX_PIXELS = 256 * 28 * 28
VIDEO_FPS = 1

# Appended to user text_input; consistent with text_output think + <answer> in JSONL (latent constrained by training template)
# FOUR_D_THINKER_TEXT_INPUT_SUFFIX = (
#     "\nReason step by step while holding a clear mental picture of the relevant object or region, "
#     "then give your answer. "
#     "Put all reasoning inside </think>...</think>, and put only the final answer inside <answer>...</answer>. "
#     "Example: </think> your reasoning (with mental imagery as needed) </think><answer> your final answer </answer>."
# )

FOUR_D_THINKER_TEXT_INPUT_SUFFIX = (
    "\nFirst, think about the reasoning process with mental imagery of the relevant object or region, "
    "then provide the user with the answer. "
    "Put the reasoning in <think>...</think> and only the final answer in <answer>...</answer>, "
    "e.g. <think> reasoning process (with mental imagery as needed) </think><answer> answer here </answer>."
)


def append_4dthinker_output_format_prompt(text_input: str) -> str:
    return text_input + FOUR_D_THINKER_TEXT_INPUT_SUFFIX

def resize_image(image, max_pixels=MAX_PIXELS):
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w = max(28, int(w * scale) // 28 * 28)
        new_h = max(28, int(h * scale) // 28 * 28)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def resize_video_frame(image, max_pixels=VIDEO_MAX_PIXELS):
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w = max(28, int(w * scale) // 28 * 28)
        new_h = max(28, int(h * scale) // 28 * 28)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def load_image(path):
    return resize_image(Image.open(path).convert("RGB"))

def load_video_frame(path):
    return resize_video_frame(Image.open(path).convert("RGB"))

def single_input_image_preprocess_function(sample):
    # Load images
    
    image = load_image(sample["image_input"])
    image_output = load_image(sample["image_output"])

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

def single_input_image_test_preprocess_function(sample):
    # Load images
    image = load_image(sample["image_input"])

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
    ]

    return conversations

def _to_file_url(p):
    """Convert path to file:// URL for qwen-vl-utils video format."""
    p = os.path.abspath(p)
    return f"file://{p}" if not p.startswith("file://") else p

def multiple_input_images_preprocess_function(sample):

    # Multiple input images: use video format with fps, consistent with training config
    user_content = []
    if isinstance(sample['image_input'], list):
        frames = [load_video_frame(p) for p in sample['image_input']]
        user_content.append({"type": "video", "video": frames, "fps": VIDEO_FPS})
    else:
        user_content.append({"type": "image", "image": load_image(sample['image_input'])})
    user_content.append({"type": "text", "text": sample["text_input"]})

    # Multiple output images: load individually as image format
    assistant_content = []
    if isinstance(sample['image_output'], list):
        for p in sample['image_output']:
            assistant_content.append({"type": "image", "image": load_image(p)})
    else:
        assistant_content.append({"type": "image", "image": load_image(sample['image_output'])})
    assistant_content.append({"type": "text", "text": sample["text_output"]})

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": assistant_content,
        },
    ]

    return conversations

def multiple_input_images_test_preprocess_function(sample):

    # Multiple input images: use video format with fps, consistent with training config
    user_content = []
    if isinstance(sample['image_input'], list):
        frames = [load_video_frame(p) for p in sample['image_input']]
        user_content.append({"type": "video", "video": frames, "fps": VIDEO_FPS})
    else:
        user_content.append({"type": "image", "image": load_image(sample['image_input'])})
    user_content.append({"type": "text", "text": sample["text_input"]})

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
    ]

    return conversations


def multiple_input_images_4dthinker_preprocess_function(sample):
    sample = dict(sample)
    sample["text_input"] = append_4dthinker_output_format_prompt(sample["text_input"])
    return multiple_input_images_preprocess_function(sample)


def multiple_input_images_4dthinker_test_preprocess_function(sample):
    sample = dict(sample)
    sample["text_input"] = append_4dthinker_output_format_prompt(sample["text_input"])
    return multiple_input_images_test_preprocess_function(sample)


task_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
    '4dthinker': multiple_input_images_4dthinker_preprocess_function,
}

task_test_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_test_preprocess_function,
    'vsp-spatial-planning': single_input_image_test_preprocess_function,
    'blink-jigsaw': multiple_input_images_test_preprocess_function,
    'sat': multiple_input_images_test_preprocess_function,
    '4dthinker': multiple_input_images_4dthinker_test_preprocess_function,
}

# Define VSP valid actions and their corresponding (row, column) changes.
ACTION_MAP = {
    "LEFT":  (0, -1),
    "DOWN":  (1,  0),
    "RIGHT": (0,  1),
    "UP":    (-1, 0),
    "L":  (0, -1),
    "D":  (1,  0),
    "R": (0,  1),
    "U":  (-1, 0)
}

def parse_action_sequence(path_str):

    path_str = path_str.upper()
    action_sequence = [char for char in list(path_str) if char in ['U', 'D', 'R', 'L']]

    return action_sequence

def simulate_vsp(map_desc, path_str):
    """
    Simulate the action sequence on the provided map.
    """
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")
    
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
        
        # If the new cell is a hole (-1), immediately return failure.
        if map_desc[new_r][new_c] == -1:
            return {
                "success": False,
                "status": "Fell in hole",
                "invalid": False,
            }

    # Check if the final position is the goal (2).
    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {
            "success": True,
            "status": "Reached goal",
            "invalid": False,
        }
    else:
        return {
            "success": False,
            "status": "Did not reach goal",
            "invalid": False,
        }