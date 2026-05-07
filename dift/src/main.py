import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import logging

from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from utils import *
from task import *
from trainer import CustomTrainerStage1

seed_everything(seed=42)
args=get_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

model_path = args.model
processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
# User-side fps is controlled by task.py content {"type": "video", "fps": ...}
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage
config.ce_loss_weight = args.ce_loss_weight
config.sim_loss_weight = args.sim_loss_weight

# DeepSpeed handles device placement; device_map="auto" is incompatible with distributed training
use_deepspeed = args.deepspeed_config is not None
model_load_kwargs = {"config": config, "torch_dtype": torch.bfloat16, "cache_dir": cache_dir}
if not use_deepspeed:
    model_load_kwargs["device_map"] = "auto"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_load_kwargs)
orig_vocab_size = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(processor.tokenizer))
# Initialize new token embeddings with the mean of existing embeddings to avoid
# random out-of-distribution values that destabilize bf16 training.
num_new_tokens = len(processor.tokenizer) - orig_vocab_size
if num_new_tokens > 0:
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        mean_embed = embed_weight[:orig_vocab_size].mean(dim=0)
        embed_weight[-num_new_tokens:] = mean_embed

        # For models with tie_word_embeddings=False (e.g. 7B), lm_head is a separate matrix,
        # new rows must be initialized separately, otherwise mismatch with embed_tokens causes NaN
        lm_head = model.get_output_embeddings()
        if lm_head is not None and not model.config.tie_word_embeddings:
            lm_head_weight = lm_head.weight
            mean_lm = lm_head_weight[:orig_vocab_size].mean(dim=0)
            lm_head_weight[-num_new_tokens:] = mean_lm

latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

for param in model.visual.parameters():
    param.requires_grad = False
model.visual.eval()
model.visual.gradient_checkpointing = False



def collate_fn_stage1(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, user_video_inputs = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, videos=user_video_inputs, return_tensors="pt", padding=True)
    del user_image_inputs, user_video_inputs

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, assistant_video_inputs = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, videos=assistant_video_inputs, return_tensors="pt", padding=True)
    del assistant_image_inputs, assistant_video_inputs

    image_inputs, video_inputs = process_vision_info(examples)
    batch = processor(text=texts, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    del image_inputs, video_inputs

    # User side supports both image (pixel_values) and video (pixel_values_videos) formats
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
    else:
        batch['pixel_values_videos'] = user_batch['pixel_values_videos']
        batch['video_grid_thw'] = user_batch['video_grid_thw']
        if 'second_per_grid_ts' in user_batch:
            batch['second_per_grid_ts'] = user_batch['second_per_grid_ts']
        # Remove leaked pixel_values from full batch (from assistant images),
        # otherwise forward will call visual encoder again to process these images
        batch.pop('pixel_values', None)
        batch.pop('image_grid_thw', None)

    # Assistant side uses image format uniformly
    batch['pixel_values_latent'] = assistant_batch['pixel_values']
    batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask

    return batch

from torch.utils.data import Dataset as TorchDataset

preprocess_function = task_preporcess_config[args.task]
raw_dataset = load_jsonl_dataset(args.data_path)

class LazyImageDataset(TorchDataset):
    def __init__(self, raw_data, preprocess_fn):
        self.raw_data = raw_data
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.preprocess_fn(self.raw_data[idx])

train_dataset = LazyImageDataset(raw_dataset, preprocess_function)
logging.info(f"Created lazy-loading dataset with {len(train_dataset)} samples")


training_args = SFTConfig(
    output_dir=args.save_model_path,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    optim="adamw_torch" if use_deepspeed else "adamw_torch_fused",
    bf16=True,
    deepspeed=args.deepspeed_config,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=["tensorboard"],
    logging_dir=args.logging_dir,
    logging_strategy='steps',
    run_name=args.exp_name,
)

# Initialize the trainer
trainer = CustomTrainerStage1(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn_stage1,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)

