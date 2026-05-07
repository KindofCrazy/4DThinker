import torch
import numpy as np
import matplotlib.pyplot as plt
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model(checkpoint_path="./models/sam3.pt")
processor = Sam3Processor(model)

# Load an image
image_path = "./example/frame_0000.jpg"
image = Image.open(image_path)
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="man")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"Masks shape: {masks.shape}")
print(f"Scores: {scores}")

# Convert image to numpy array
image_np = np.array(image)

# Create a figure to render the mask on the original image
plt.figure(figsize=(10, 10))
plt.imshow(image_np)

# Render masks on the image
if len(masks) > 0:
    # Take the first mask (highest score)
    mask = masks[0].cpu().numpy() if torch.is_tensor(masks[0]) else masks[0]
    
    # Squeeze extra dimensions and ensure it's 2D
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    
    print(f"Processed mask shape: {mask.shape}")
    
    # Create a colored mask overlay (semi-transparent)
    # mask should now be 2D (H, W)
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask > 0] = [1, 0, 0, 0.5]  # Red color with 50% transparency
    
    plt.imshow(colored_mask)
    
    # Draw bounding box if available
    if boxes is not None and len(boxes) > 0:
        box = boxes[0].cpu().numpy() if torch.is_tensor(boxes[0]) else boxes[0]
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          fill=False, edgecolor='green', linewidth=2))

plt.axis('off')
plt.tight_layout()

# Save the rendered image
output_path = "./example/frame_0000_masked.jpg"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
print(f"Rendered image saved to: {output_path}")
plt.close()

# #################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]