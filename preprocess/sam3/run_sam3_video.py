import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(checkpoint_path="./models/sam3.pt")
video_path = "./example/video_frames" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="man",
    )
)
output = response["outputs"]
print(output)
print(output["out_binary_masks"].shape)