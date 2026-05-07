from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import os
from datetime import datetime
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import ast
import os
from openai import OpenAI
from transformers.utils.versions import require_version
import json
import base64
from open_r1.trainer.record import reward_record
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import math
import sys
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import re

def get_first_word(text):
    match = re.search(r'\b\w+\b', text)
    return match.group() if match else ""

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

def extract_option(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.
    
    Args:
        text: The model response text
        
    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found
    """
    if not text:
        return None
    
    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r'([A-E])\.', text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)
    
    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r'<Answer>(.*?)(?:<|$)', text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [
            r'[Mm]y answer is ([A-E])',
            r'[Mm]y answer is ([A-E])\.',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer: )?([A-E])\.',
            r'\b([A-E])\b'
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)
    
    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',  # Answer at line beginning
        r'[\*\"]([A-E])[\*\"]',  # Answer in quotes or asterisks
        r'\bAnswer:?\s*([A-E])\b',  # Answer following "Answer:"
        r'[Mm]y answer is ([A-E])',  # Added pattern for "My answer is X"
        r'[Mm]y answer is ([A-E])\.',  # Added pattern for "My answer is X."
        r'answer is ([A-E])',  # Added pattern for phrases like "The answer is X"
    ]
    
    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)
    
    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split('\n')
    line_matches = []
    
    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))
    
    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]
    
    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    
    return None  # No answer found

def encode_image(image_path):
    # image_path = image_path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_answer(xml_text):
    # Use regex to match content between <answer> and </answer> tags, including newlines
    pattern = r'<answer>\n?(.*?)\n?</answer>'  # Handle possible newlines
    match = re.search(pattern, xml_text, re.DOTALL)
    
    if match:
        # Extract content and strip whitespace
        return match.group(1).strip()
    return ""

# def judge_score_func(client, question, gt_response, pred_response, base64_image):
#     question = question[0]['content'][-1]['text'].replace(' First think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.','')
#     question = question.replace(' The first image serves as the main image, followed by four generated images that show different perspectives. The last one is the depth image corresponding to the main image.','')
#     question = question.replace(' The first image serves as the main image, followed by four generated images that show different perspectives.','')
#     question = question.replace(' The first image serves as the main image, followed by the depth image corresponding to the main image.','')
#     SYSTEM_PROMPT = f'''You are responsible for proofreading the answers, you need to give the score to the model's answer by referring to the standard answer, based on the given question and image.
#     The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
#     The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
#     '''
#     PROMPT = f'''
#     Question: {question}
#     Standard answer: {gt_response}
#     Model's answer: {pred_response}
#     '''
#     messages_list = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": PROMPT},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#             ],
#         }]
    
#     result = client.chat.completions.create(
#         model="gpt-4o-2024-11-20",
#         messages=messages_list,
#         stream=False,
#         extra_headers={
#             "M-TraceId": "2136218312678"
#         }
#     )
#     response = result.choices[0].message.content
#     return response

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        ## todo
        match task_type:
            case "reward_model":
                return "{Question}\nIs image 2 better than image 1? Please answer \"yes\" or \"no\", and given the detailed reason. The reasoning process is enclosed within <think> </think> tags, i.e., Yes/No. <think> reasoning process here </think>. If you cannot compare the two images, answer \"Null\"."
            case "rec":
                return "{Question}\nFirst think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Special tokens should be used to represent 3D imagery during the reasoning process, i.e., <think> reasoning process with special tokens here </think><answer> answer here </answer>."
                # return "{Question} First think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
            case "ic":
                return "{Question} First think about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
            case "odLength":
                SYSTEM_PROMPT = (
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    
    
    ## TODO: Added latent validation logic   
    # @staticmethod
    # def format_reward(completions, **kwargs):
    #     # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    #     pattern = r"<think>.*?<\|latent_start\|>(<\|latent_pad\|>)+<\|latent_end\|>.*?</think>\s*<answer>.*?</answer>"
    #     completion_contents = [completion[0]["content"] for completion in completions]
    #     matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
    #     rewards = [1.0 if match else 0.0 for match in matches]
    #     print("--------Format Reward--------")
    #     for content, reward in zip(completion_contents, rewards):
    #         if reward == 1.0:
    #             print(f"Matching Content: {content}")
    #     print(f"Reward: {rewards}")
    #     return rewards
    
    @staticmethod
    def format_reward(completions, **kwargs):
        # Match content starting with yes or no (case insensitive)
        pattern = r"^\s*(yes|no)\b.*<think>.*?</think>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content.strip(), re.DOTALL | re.IGNORECASE) is not None for content in completion_contents]
        rewards = [1.0 if match else 0.0 for match in matches]
        print("--------Format Reward--------")
        # for content, reward in zip(completion_contents, rewards):
        #     if reward == 1.0:
        #         print(f"Matching Content: {content}")
        print(f"Reward: {rewards}")
        return rewards

    # @staticmethod
    # def response_text_reward(completions, answer, prompts, img_path_str, **kwargs):
    #     contents = [completion[0]["content"] for completion in completions]
    #     rewards = []
    #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    #     for content, gt_response, prompt, img in zip(contents, answer, prompts,img_path_str):
    #         reward = 0.0
    #         try:
    #             # base64_image = encode_image(img)
    #             pred_response = extract_answer(content)
    #             if pred_response.strip().lower() in gt_response.strip().lower():
    #                 reward = 1.0
    #         except Exception:
    #              print("service error")
    #              pass  # Continue to next verification method if this fails

    #         print("--------Response Reward--------")
    #         print(f"Question:{prompt[0]['content'][-1]['text']}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")

    #         rewards.append(reward)
    #         if os.getenv("DEBUG_MODE") == "true":
    #             log_path = os.getenv("LOG_PATH")
    #             with open(log_path, "a", encoding='utf-8') as f:
    #                 f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
    #                 f.write(f"Content: {content}\n")
    #                 f.write(f"Answer: {gt_response}\n")
    #     return rewards
    
    @staticmethod
    def response_text_reward(completions, answer, prompts, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, gt_response, prompt in zip(contents, answer, prompts):
            reward = 0.0
            try:
                if gt_response == True:
                    gt_answer = "yes"
                else:
                    gt_answer = "no"
                pred_response = get_first_word(content)
                if pred_response.strip().lower() == gt_answer.strip().lower():
                    reward = 1.0
            except Exception:
                gt_answer = ""
                pred_response = ""
                print("service error")
                pass  # Continue to next verification method if this fails

            print("--------Response Reward--------")
            # print(f"Question:{prompt[0]['content'][-1]['text']}, content: {content}, GT:{gt_response}, Pred:{pred_response}, Response Accuracy: {reward}")
            print(f"Question:{prompt[0]['content'][-1]['text']}, GT:{gt_answer}, Pred:{pred_response}, Response Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {gt_response}\n")
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "format":
            match task_type:
                case "reward_model":
                    return Qwen2VLModule.format_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "response":
            match task_type:
                case "reward_model":
                    return Qwen2VLModule.response_text_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")