from trl import SFTTrainer, SFTConfig
import torch
import re
import logging

class CustomTrainerStage1(SFTTrainer):

    def _debug_decode_sample(self, inputs, logits, loss, ce_loss, sim_loss, num_images):
        """Decode and log a training sample for debugging.
        Args:
            logits: detached logits tensor [seq_len, vocab_size] for sample 0.
        """
        try:
            sample_input_ids = inputs["input_ids"][0]
            sample_labels = inputs["labels"][0]

            input_text = self.tokenizer.decode(sample_input_ids, skip_special_tokens=False)
            question_part = input_text
            if "<|im_start|>assistant" in input_text:
                question_part = input_text.split("<|im_start|>assistant")[0]
                if "<|im_start|>user" in question_part:
                    question_part = question_part.split("<|im_start|>user")[-1]
            question_part = re.sub(
                r"<\|vision_start\|>.*?<\|vision_end\|>", "<video>", question_part, flags=re.DOTALL
            )
            question_part = question_part.strip()[:300]

            valid_mask = sample_labels != -100
            gt_ids = sample_labels[valid_mask]
            gt_text = self.tokenizer.decode(gt_ids, skip_special_tokens=False)
            gt_text = re.sub(r"<\|latent_start\|>", "", gt_text)
            gt_text = re.sub(r"<\|latent_end\|>", "", gt_text)
            gt_text = re.sub(r"<\|latent_pad\|>", "", gt_text)
            gt_text = re.sub(r"\s+", " ", gt_text).strip()[:500]

            if logits is not None:
                shift_logits = logits[:-1]
                shift_labels = sample_labels[1:]
                shift_valid = shift_labels != -100
                if shift_valid.any():
                    pred_ids = shift_logits[shift_valid].argmax(dim=-1)
                    pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=False)
                    pred_text = re.sub(r"<\|latent_start\|>", "", pred_text)
                    pred_text = re.sub(r"<\|latent_end\|>", "", pred_text)
                    pred_text = re.sub(r"<\|latent_pad\|>", "", pred_text)
                    pred_text = re.sub(r"\s+", " ", pred_text).strip()[:500]
                else:
                    pred_text = "<no valid predictions>"
            else:
                pred_text = "<logits not available>"

            logging.info(
                f"\n{'='*60}\n"
                f"[DEBUG Step {self.state.global_step}]\n"
                f"  total_loss={loss.item():.4f}  ce_loss={ce_loss.item():.4f}  "
                f"sim_loss={sim_loss.item():.4f}  num_images={num_images}\n"
                f"  seq_len={sample_input_ids.shape[0]}\n"
                f"  Question : {question_part}\n"
                f"  GT Answer: {gt_text}\n"
                f"  Pred     : {pred_text}\n"
                f"{'='*60}"
            )
        except Exception as e:
            logging.warning(f"[DEBUG] Failed to decode at step {self.state.global_step}: {e}")

    def _make_zero_loss(self, model, inputs, dtype):
        """Construct zero loss (keeping computation graph connected) to skip current batch."""
        underlying = model.module if hasattr(model, 'module') else model
        for p in underlying.parameters():
            if p.requires_grad:
                return (p.flatten()[0] * 0.0).to(dtype)
        return torch.tensor(0.0, device=inputs["input_ids"].device, dtype=dtype)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ====== Ensure visual encoder is in eval mode with gradient checkpointing disabled ======
        underlying = model.module if hasattr(model, 'module') else model
        underlying.visual.eval()
        underlying.visual.gradient_checkpointing = False

        # ====== NaN diagnosis: check model weights for NaN before forward ======
        if self.state.global_step <= 5:
            with torch.no_grad():
                nan_params = []
                for name, p in underlying.named_parameters():
                    if torch.isnan(p).any():
                        nan_params.append(name)
                    elif torch.isinf(p).any():
                        nan_params.append(f"{name}(Inf)")
                if nan_params:
                    logging.warning(
                        f"[Step {self.state.global_step}] NaN/Inf in model params BEFORE forward: "
                        f"{nan_params[:10]}{'...' if len(nan_params) > 10 else ''} "
                        f"(total {len(nan_params)} params)"
                    )

        # ====== OOM protection: forward may OOM on long sequences, skip this batch ======
        try:
            (ce_loss, outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        except torch.cuda.OutOfMemoryError:
            seq_len = inputs["input_ids"].shape[-1]
            logging.warning(
                f"[Step {self.state.global_step}] OOM during forward (seq_len={seq_len}), "
                f"skipping this batch"
            )
            torch.cuda.empty_cache()
            zero_loss = self._make_zero_loss(model, inputs, torch.bfloat16)
            return (zero_loss, None) if return_outputs else zero_loss

        # ====== NaN protection: skip batches that produce NaN loss ======
        if torch.isnan(ce_loss):
            logging.warning(
                f"[Step {self.state.global_step}] NaN ce_loss detected, skipping this batch"
            )
            del outputs
            zero_loss = self._make_zero_loss(model, inputs, ce_loss.dtype)
            return (zero_loss, None) if return_outputs else zero_loss

        # Extract needed tensors then release outputs immediately to avoid holding
        # full logits/hidden_states during gradient accumulation causing linear memory growth
        predict_embeddings = outputs.hidden_states
        input_embeddings = outputs.inputs_embeds
        # Only take detached logits for decoding when debugging (does not prevent memory release)
        should_log = (self.state.global_step % self.args.logging_steps == 0
                      and self.args.local_rank in [-1, 0])
        debug_logits = outputs.logits[0].detach() if should_log else None
        del outputs

        image_out_mask = inputs["image_out_mask"]

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()
        del predict_embeddings

        # GT embeddings: .detach() is CRITICAL — gt is a fixed target, must not backprop
        # through inputs_embeds → embed_tokens graph.
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous().detach()
        del input_embeddings

        cfg = underlying.config
        latent_size = cfg.latent_size
        ce_weight = getattr(cfg, 'ce_loss_weight', 0.1)
        sim_weight = getattr(cfg, 'sim_loss_weight', 1.0)

        num_latent_tokens = shift_predict_embeddings.shape[0]
        num_images = num_latent_tokens // latent_size

        if num_images == 0:
            sim_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
            loss = ce_weight * ce_loss
        else:
            pred_per_image = shift_predict_embeddings.view(num_images, latent_size, -1)
            gt_per_image = gt_embeddings.view(num_images, latent_size, -1)

            # Normalize-then-dot in float32: equivalent to cosine similarity but with
            # bounded gradients (no 1/||x||^3 term), preventing NaN in bf16 training.
            pred_norm = torch.nn.functional.normalize(pred_per_image.float(), dim=-1)
            gt_norm = torch.nn.functional.normalize(gt_per_image.float(), dim=-1)
            cos_sim_per_token = (pred_norm * gt_norm).sum(dim=-1)

            sim_loss_per_image = (1 - cos_sim_per_token).mean(dim=1)
            sim_loss = sim_loss_per_image.mean()
            loss = ce_weight * ce_loss + sim_weight * sim_loss

        del shift_predict_embeddings, gt_embeddings

        # ====== Final NaN fallback (sim_loss or combined loss may be NaN) ======
        if torch.isnan(loss):
            logging.warning(
                f"[Step {self.state.global_step}] NaN total loss detected (ce={ce_loss.item()}, "
                f"sim={sim_loss.item()}), skipping this batch"
            )
            loss = self._make_zero_loss(model, inputs, ce_loss.dtype)

        if should_log:
            self.log({
                "train/total_loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/sim_loss": sim_loss.item(),
                "train/cos_similarity": (1 - sim_loss).item(),
                "train/num_output_images": num_images,
            })
            self._debug_decode_sample(inputs, debug_logits, loss, ce_loss, sim_loss, num_images)
            del debug_logits

        # Do not return outputs, avoid Trainer holding references during gradient accumulation
        return (loss, None) if return_outputs else loss
