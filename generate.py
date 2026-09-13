from contextlib import contextmanager
from pathlib import Path
from pydantic import BaseModel, TypeAdapter
from threading import Thread

import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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

        self.device = 'cuda'
        self.batch_size = 1
        self.grad_accumulation_step = 8
        self.context_length = 1024
        self.checkpoint_count = 500
        self.last_processed_entry = 0

def tokenize_chat(tokenizer, chats, loss_start, loss_end):
    tokenized = tokenizer.apply_chat_template(
            chats, tokenize = True, 
            add_special_tokens = True, return_tensors='pt', 
            add_generation_prompt = True,
            padding = True
    ) 
    chat_text      = tokenized ['input_ids']
    attention_mask = tokenized ['attention_mask']

    loss_start_idx = (chat_text == loss_start)
    loss_end_idx   = (chat_text == loss_end)

    loss_application_mask = (loss_start_idx.long() - loss_end_idx.long()).cumsum(dim = -1).to(torch.bool)
    loss_idx_removal_mask = (loss_start_idx | loss_end_idx).to(torch.bool)

    loss_application_mask[loss_idx_removal_mask] = False
    attention_mask[loss_idx_removal_mask] = 0
    labels = chat_text.clone()
    labels[~loss_application_mask] = -100
    labels[attention_mask == 0] = -100

    return chat_text, attention_mask, labels

@contextmanager
def checkpoint_manager(
        tokenizer, model, optimizer, scheduler, scaler,
        run_id, device, checkpoint_dir, load_checkpoint = True):
    
    directory = Path(checkpoint_dir)
    directory.mkdir(parents = True, exist_ok = True)

    context = Context()
    context.tokenizer = tokenizer
    context.model = model
    context.optimizer = optimizer
    context.scaler = scaler
    context.scheduler = scheduler
    context.device = device


    def do_load_checkpoint(filename):
        filename = directory / filename
        checkpoint = torch.load(filename, weights_only=True)
        if model is not None: model.load_state_dict(checkpoint['model'])
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    latest_file  = None
    latest_time  = None
    for file_stat in directory.glob("*.pt"):
        mod_time = file_stat.stat().st_mtime
        if latest_time == None or mod_time > latest_time:
            latest_file  = file_stat.name
            latest_time  = mod_time
    if load_checkpoint and latest_file is not None:
        print("Loading checkpoint")
        do_load_checkpoint(latest_file)
    else:
        print("Skipping checkpoint")

    try:
        yield context
    finally:
        pass

def build_model(model_dir, device, peft=True):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    specialize_tokenizer(tokenizer, './generate.jinja')

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16) # , attn_implementation="flash_attention_2") 
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()
    model.config.use_cache = True

    if peft:
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


def generate(
        tokenizer, model, 
        checkpoint_dir, 
        device,
        samples = 5,
        checkpoint = True,
        data = {},
    ): 


    loss_start = tokenizer('<|loss_start|>', return_tensors ='pt')['input_ids']
    loss_end = tokenizer('<|loss_end|>', return_tensors ='pt')['input_ids']
    assert loss_start.numel() == 1, f"loss_start tokenized to {loss_start.numel()} tokens"
    assert loss_end.numel() == 1, f"loss_end tokenized to {loss_end.numel()} tokens"

    loss_start = loss_start[0]
    loss_end = loss_end[0]


    input_ids, attention_mask, labels = tokenize_chat(tokenizer, [data], loss_start, loss_end)
    inputs = {
            'input_ids': input_ids.to(model.device),
            'attention_mask': attention_mask.to(model.device),
            # 'labels': labels,
    }

    with checkpoint_manager(tokenizer, model, None, None, None, None, device, checkpoint_dir, load_checkpoint=checkpoint) as ctx:
        streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=False, skip_special_tokens=False,
        )

        for i in range(samples):
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 250,
                "streamer": streamer,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": True,
                "top_p": 0.95,
            }

            thread = Thread(
                    target = lambda **x: model.generate(**x),
                    kwargs = generation_kwargs,
            )
            thread.start()

            for text in streamer:
                print(text, end="", flush=True)
            thread.join()

if __name__ == "__main__":
    import sys
    print(len(sys.argv))
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} <model> <checkpoint_dir>")
        exit()

    device = "cuda"

    data = [
            {'role': 'user', 'content': "Hello how are you"},
    ]

    with torch.inference_mode():
        print("Loading tokenizer and model")
        tokenizer, sftModel= build_model(
            model_dir = sys.argv[1], # "./model/SmolLM-135M",
            device = device,
            peft = True,
        )

        print("Generating")
        generate(tokenizer, sftModel, 
            checkpoint_dir = sys.argv[2], # "./model/checkpoint/",
            device = device, checkpoint = True, data = data)

        del sftModel
