import torch
import splitData
from transformers import AutoTokenizer, MBartForSequenceClassification
from torch.utils.data import RandomSampler, SequentialSampler
from utils import make_dataset, get_dataloader, run

train, test, valid = splitData.eng_curated()

epochs = 5
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
print(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, SequentialSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset,SequentialSampler, batch_size)

print(train_dataset[0])

model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25", num_labels = 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

run(model, "mBERT", optimizer, train_dataloader, valid_dataloader, test_dataloader, epochs)