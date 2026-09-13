import sys
import json

from pydantic import BaseModel, TypeAdapter
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import pdb

from tokenizer import specialize_tokenizer

class Conversation(BaseModel):
    role: str
    content: str
conversation = TypeAdapter(list[Conversation])

BATCH_SIZE = 500
SKIP = 0

def tokenize_chat(tokenizer, chats, loss_start, loss_end):
    tokenized = tokenizer.apply_chat_template(
            chats, tokenize = True, 
            add_special_tokens = True, return_tensors='pt', 
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

def main():
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
    specialize_tokenizer(tokenizer)

    loss_start = tokenizer('<|loss_start|>', return_tensors ='pt')['input_ids']
    loss_end = tokenizer('<|loss_end|>', return_tensors ='pt')['input_ids']
    assert loss_start.numel() == 1, f"loss_start tokenized to {loss_start.numel()} tokens"
    assert loss_end.numel() == 1, f"loss_end tokenized to {loss_end.numel()} tokens"

    loss_start = loss_start[0]
    loss_end = loss_end[0]

    dataset = load_dataset(sys.argv[1], streaming=True, split='train')
    dataset = dataset.skip(SKIP)

    chats = []

    def save_chats(chats, idx):
        chat_text, attention_mask, labels = tokenize_chat(tokenizer, chats, loss_start, loss_end)
        ds = Dataset.from_dict({
            'input_ids': chat_text,
            'labels': labels,
            'attention_mask': attention_mask,
        })
        ds.save_to_disk(f'dataset/train/tokenized_{idx}.bin')

    for idx, training in enumerate(dataset):
        idx += SKIP
        chat = build_chat(training)
        chats.append(chat)

        if len(chats) == BATCH_SIZE:
            save_chats(chats, idx)
            chats = []

    if len(chats) > 0:
       save_chats(chats, idx)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} <dataset_loc> <tokenizer>")
        exit()

    main()
