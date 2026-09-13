import sys
import atexit
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F 
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime, date
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset


class Context:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = None

        self._directory = None

        self.checkpoint_count = 3
        self.last_processed_entry   = 0
        self.batch_size             = 1
        self.grad_accumulation_step = 1

        self.use_scaler = None

        assert self.checkpoint_count % self.grad_accumulation_step == 0

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, name):
        self._checkpoint_dir = name
        self._directory = Path(self.checkpoint_dir)
        self._directory.mkdir(parents = True, exist_ok = True)

    def cleanup(self, checkpoint_to_save = 2):
        checkpoints = sorted(
            self._directory.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        files_to_delete = checkpoints[:-checkpoint_to_save]
        for rm_file in files_to_delete:
            print(f'Removing checkpoint: {rm_file}')
            os.remove(rm_file)
    

    def load_latest_checkpoint(self):
        latest_file  = None
        latest_time  = None
        for file_stat in self._directory.glob("*.pt"):
            mod_time = file_stat.stat().st_mtime
            if latest_time == None or mod_time > latest_time:
                latest_file  = file_stat.name
                latest_time  = mod_time
        if latest_file is not None:
            self.load_checkpoint(latest_file)


    def save_checkpoint(self, filename):
        print(f"Saving Checkpoint {filename}")
        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(to_cpu(v) for v in obj)
            else:
                return obj
        filename = self._directory / filename
        # Move state dicts to CPU before saving to free GPU memory during serialization
        model_state = {k: to_cpu(v) for k, v in self.model.state_dict().items()}
        optimizer_state = {k: to_cpu(v) for k, v in self.optimizer.state_dict().items()}
        scheduler_state = {k: to_cpu(v) for k, v in self.scheduler.state_dict().items()}
        scaler_state = {k: to_cpu(v) for k, v in self.scaler.state_dict().items()}
        state = {
                'model': model_state,
                'optimizer': optimizer_state,
                'scheduler': scheduler_state,
                'scaler': scaler_state,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'last_processed_entry': self.last_processed_entry,
        }
        pt = filename.with_suffix('.pt')
        tmp = filename.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, pt)

    def load_checkpoint(self, filename):
        print(f"Loading Checkpoint {filename}")
        filename = self._directory / filename
        checkpoint = torch.load(filename, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if 'last_processed_entry' in checkpoint:
            self.last_processed_entry = checkpoint['last_processed_entry']

def read_dataset(ctx, dataset_dir):
    dataset = load_dataset(dataset_dir, streaming=True, split='train')
    dataset = dataset.skip(ctx.last_processed_entry)

    def process(input_val):
        chats = []
        last_chat = None
        input_vals = input_val.split("\n\n")
        for v in input_vals:
            if len(v) == 0:
                continue

            if v.startswith("Human:"):
                v = v.removeprefix("Human:")
                role = 'user'
                content = v.strip()
            elif v.startswith("Assistant:"):
                v = v.removeprefix("Assistant:")
                role = 'assistant'
                content = v.strip()
            else:
                last_chat['content'] += "\n\n" + v
                continue

            last_chat = {
                'role': role,
                'content': content,
            }
            chats.append(last_chat)

        return chats

    max_length = ctx.tokenizer.model_max_length

    inputs = []
    items_processed = 0
    for idx, data in enumerate(dataset):
        chosen, rejected = data['chosen'], data['rejected']
        inputs.append(process(chosen))
        inputs.append(process(rejected))
        items_processed += 1

        if len(inputs) < 2 * ctx.batch_size:
            continue

        response = ctx.tokenizer.apply_chat_template(
                inputs, 
                add_special_tokens = True,
                tokenize = True,
                return_tensors="pt", 
                padding = True, truncation=True,
                max_length = max_length
            ).to(ctx.device)
        ctx.last_processed_entry += items_processed
        items_processed = 0
        inputs = []
        yield response



def load_tokenizer_and_model(model_dir, template_file, device):
    with open(template_file) as f:
        template = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = template
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype = torch.bfloat16)

    head_in_features = model.lm_head.in_features
    model.lm_head = nn.Linear(head_in_features, 1, bias = False, dtype = torch.bfloat16)
    model.lm_head.requires_grad_(True)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
            ],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)

    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()


    model.print_trainable_parameters()
    return tokenizer, model


def main(ctx, dataset):
    model = ctx.model
    ctx.optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr = 2e-5,
    )

    ctx.scheduler = CosineAnnealingLR(
            ctx.optimizer,
            T_max = 100_000,
            eta_min=2e-5,
    )
    ctx.scaler = GradScaler(ctx.device)
    ctx.load_latest_checkpoint()

    def save_final_checkpoint():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"exit_checkpoint_{timestamp}"
        ctx.save_checkpoint(run_id)
        ctx.cleanup()

    atexit.register(save_final_checkpoint)

    for data in dataset:
        with torch.autocast(dtype = torch.bfloat16, device_type = ctx.device):
            response = model(**data)
            scores   = response.logits
            
            chosen_scores = scores[0::2, -1, :].squeeze(-1)
            rejected_scores = scores[1::2, -1, :].squeeze(-1)
            diff = chosen_scores - rejected_scores
            # loss = -log(sigmoid(diff))
            # sigmoid(diff) = 1 / (1 + exp(-diff))
            # loss = -(-log(1 + exp(-diff)))
            # loss = log(1 + exp(-diff)) = softplus(-diff)
            loss = F.softplus(-diff).mean()
            loss /= ctx.grad_accumulation_step

        if ctx.use_scaler:
            ctx.scaler.scale(loss).backward()
        else:
            loss.backward()

        print(ctx.last_processed_entry, loss.item())
        if ctx.last_processed_entry % ctx.grad_accumulation_step == 0:
            if ctx.use_scaler:
                ctx.scaler.step(ctx.optimizer)
                ctx.scaler.update()
            else:
                ctx.optimizer.step()
            ctx.scheduler.step()
            ctx.optimizer.zero_grad(set_to_none = True)
            
        if ctx.last_processed_entry % ctx.checkpoint_count == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"checkpoint_{timestamp}"
            ctx.save_checkpoint(run_id)
            ctx.cleanup()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"{sys.argv[0]} <model_dir> <checkpoint_dir> <dataset_dir> <template_file>")
        exit()

    device = "cuda"

    model_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    dataset_dir = sys.argv[3]
    template_file = sys.argv[4]
    tokenizer, model = load_tokenizer_and_model(model_dir, template_file, device)

    ctx = Context(tokenizer, model)
    ctx.checkpoint_dir = checkpoint_dir
    ctx.device = device
    ctx.use_scaler = True

    dataset = read_dataset(ctx, dataset_dir)
    main(ctx, dataset)
