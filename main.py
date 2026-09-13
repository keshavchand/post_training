import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

import torch
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenizer import specialize_tokenizer

class Conversation(BaseModel):
    role: str
    content: str
conversation = TypeAdapter(list[Conversation])

class Context:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_start = None
        self.loss_end = None

        self.device = 'cuda'
        self.batch_size = 1
        self.grad_accumulation_step = 8
        self.context_length = 1024
        self.checkpoint_count = 500
        self.last_processed_entry = 0

def build_chat(training):
    chats = []
    #training = json.loads(training['messages'])
    training = conversation.validate_json(training['messages'])
    for data in training:
        chat = {
                'role': data.role,
                'content': data.content,
        }
        if data.role == 'assistant':
            chat['loss_token'] = True
        chats.append(chat)

    return chats

def tokenize_chat(tokenizer, chat, loss_start, loss_end):
    tokenized = tokenizer.apply_chat_template(
            chat, tokenize = True, 
            add_special_tokens = True, return_tensors='pt', 
            padding = True
    ) 
    input_ids      = tokenized ['input_ids']
    attention_mask = tokenized ['attention_mask']

    loss_start_idx = (input_ids == loss_start)
    loss_end_idx   = (input_ids == loss_end)

    loss_application_mask = (loss_start_idx.long() - loss_end_idx.long()).cumsum(dim = -1).to(torch.bool)
    loss_idx_removal_mask = (loss_start_idx | loss_end_idx).to(torch.bool)

    loss_application_mask[loss_idx_removal_mask] = False
    attention_mask[loss_idx_removal_mask] = 0
    labels = input_ids.clone()
    labels[~loss_application_mask] = -100
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels

def train_model(ctx: Context, ds: Dataset):
    scaler = ctx.scaler
    optimizer = ctx.optimizer
    scheduler = ctx.scheduler
    optimizer.zero_grad(set_to_none = True)
    chats = []
    for idx, data in enumerate(ds):
        ctx.last_processed_entry += 1
        torch.cuda.reset_peak_memory_stats()

        chats.append(build_chat(data))
        if len(chats) < ctx.batch_size:
            continue

        input_ids, attention_mask, labels = tokenize_chat(
                ctx.tokenizer, 
                chats, 
                ctx.loss_start, 
                ctx.loss_end)

        chats = []

        input_ids = input_ids[:, :ctx.context_length].to(ctx.device, non_blocking = True)
        labels = labels[:, :ctx.context_length].to(ctx.device, non_blocking = True)
        attention_mask = attention_mask[:, :ctx.context_length].to(ctx.device, non_blocking = True)

        print(f"step {idx}: input={input_ids.shape}, allocated={torch.cuda.memory_allocated()/1024**2:.0f} MB")

        with torch.autocast(ctx.device, torch.float16):
            output = ctx.model(
                    input_ids = input_ids,
                    labels = labels,
                    attention_mask = attention_mask,
            )
            loss = output.loss / ctx.grad_accumulation_step

        # Free input tensors once the forward pass is done to lower peak memory
        del input_ids, labels, attention_mask
        scaler.scale(loss).backward()
        if (( idx + 1 ) % ctx.grad_accumulation_step == 0):
            print(f"Run {idx}:", loss.item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none = True)

        if idx != 0 and idx % ctx.checkpoint_count == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"checkpoint_{timestamp}"
            ctx.save_checkpoint(run_id)
            ctx.cleanup()

        del output, loss
        print(f"step {idx}, allocated={torch.cuda.memory_allocated()/1024**2:.0f} MB")


@contextmanager
def checkpoint_manager(
        tokenizer, model, optimizer, scheduler, scaler,
        run_id, device, checkpoint_dir):
    
    directory = Path(checkpoint_dir)
    directory.mkdir(parents = True, exist_ok = True)

    context = Context()
    context.tokenizer = tokenizer
    context.model = model
    context.optimizer = optimizer
    context.scaler = scaler
    context.scheduler = scheduler
    context.device = device

    def cleanup(checkpoint_to_save = 2):
        checkpoints = sorted(
            directory.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        files_to_delete = checkpoints[:-checkpoint_to_save]
        for rm_file in files_to_delete:
            print(f'Removing checkpoint: {rm_file}')
            os.remove(rm_file)
    
    def save_checkpoint(filename):
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
        filename = directory / filename
        # Move state dicts to CPU before saving to free GPU memory during serialization
        model_state = {k: to_cpu(v) for k, v in model.state_dict().items()}
        optimizer_state = {k: to_cpu(v) for k, v in optimizer.state_dict().items()}
        scheduler_state = {k: to_cpu(v) for k, v in scheduler.state_dict().items()}
        scaler_state = {k: to_cpu(v) for k, v in scaler.state_dict().items()}
        state = {
                'model': model_state,
                'optimizer': optimizer_state,
                'scheduler': scheduler_state,
                'scaler': scaler_state,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'last_processed_entry': context.last_processed_entry,
        }
        pt = filename.with_suffix('.pt')
        tmp = filename.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, pt)



    def load_checkpoint(filename):
        filename = directory / filename
        checkpoint = torch.load(filename, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if 'last_processed_entry' in checkpoint:
            context.last_processed_entry = checkpoint['last_processed_entry']

    context.save_checkpoint = save_checkpoint
    context.cleanup = cleanup 

    latest_file  = None
    latest_time  = None
    for file_stat in directory.glob("*.pt"):
        mod_time = file_stat.stat().st_mtime
        if latest_time == None or mod_time > latest_time:
            latest_file  = file_stat.name
            latest_time  = mod_time
    if latest_file is not None:
        load_checkpoint(latest_file)

    try:
        yield context
    finally:
        save_checkpoint(f'{run_id}')
        cleanup()

def build_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    specialize_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16) # , attn_implementation="flash_attention_2") 
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
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


def main(
        tokenizer, model, 
        checkpoint_dir, 
        dataset_name,
        device,
    ): 
    optimizer= AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=2e-4, weight_decay = 0.01)
    scheduler = CosineAnnealingLR(
            optimizer,
            T_max = 10_000,
            eta_min=2e-5,
        )


    loss_start = tokenizer('<|loss_start|>', return_tensors ='pt')['input_ids']
    loss_end = tokenizer('<|loss_end|>', return_tensors ='pt')['input_ids']
    assert loss_start.numel() == 1, f"loss_start tokenized to {loss_start.numel()} tokens"
    assert loss_end.numel() == 1, f"loss_end tokenized to {loss_end.numel()} tokens"


    loss_start = loss_start[0]
    loss_end = loss_end[0]

    run_id = f"checkpoint_{time.time()}"
    scaler = GradScaler(device)
    
    with checkpoint_manager(tokenizer, model, optimizer, scheduler, scaler, run_id, device, checkpoint_dir) as ctx:
        ctx.loss_start = loss_start
        ctx.loss_end = loss_end
        dataset = load_dataset(dataset_name, streaming=True, split='train')
        dataset = dataset.skip(ctx.last_processed_entry)
        train_model(ctx, dataset)

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) != 4:
        print(f"{sys.argv[0]} <model> <checkpoint_dir> <dataset_name>")
        exit()

    device = "cuda"
    print("Loading tokenizer and model")
    tokenizer, model = build_model(
        model_dir = sys.argv[1], # "./model/SmolLM-135M",
        device = device)

    print("Starting Training")
    main(tokenizer, model, 
        checkpoint_dir = sys.argv[2], # "./model/checkpoint/",
        dataset_name = sys.argv[3],
        device = device)
