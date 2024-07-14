import torch
import splitData
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import RandomSampler, SequentialSampler
from utils import make_dataset, get_dataloader, run

train, test, valid = splitData.kor_unsmile()

epochs = 5
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, SequentialSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset,SequentialSampler, batch_size)

print(train_dataset[0])

model = BertModel.from_pretrained("skt/kobert-base-v1").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

run(model, "koBERT", optimizer, train_dataloader, valid_dataloader, test_dataloader, epochs)