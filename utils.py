import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.metrics import classification_report

def make_dataset(data, tokenizer, device):
  text_key = "문장"
  label_key = "clean"
  try:
    data["문장"]
  except KeyError:
    text_key = "Content"
    label_key = "Label"
  
  tokenized = tokenizer(
      text=data[text_key].tolist(),
      padding="longest",
      truncation=True,
      return_tensors="pt"
  )
  input_ids = tokenized["input_ids"].to(device)
  attention_mask = tokenized["attention_mask"].to(device)
  labels = torch.tensor([(1 if val == 0 else 0) if label_key=="clean" else val for val in data[label_key]], dtype=torch.long).to(device)
  return TensorDataset(input_ids, attention_mask, labels)

def get_dataloader(dataset, sampler, batch_size):
  data_sampler = sampler(dataset)
  dataloader = DataLoader(dataset, sampler=data_sampler, batch_size = batch_size)
  return dataloader

def get_flats(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten().tolist()
  labels_flat = labels.flatten().tolist()
  return (pred_flat, labels_flat)

def train(model, optimizer, dataloader):
  model.train()
  train_loss = 0.0
  for input_ids, attention_mask, labels in dataloader:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss = train_loss / len(dataloader)
  return train_loss

def evaluation(model, dataloader):
  with torch.no_grad():
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    pred_flat_sum = []
    labels_flat_sum = []

    for input_ids, attention_mask, labels in dataloader:
      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      logits = outputs.logits

      loss = criterion(logits, labels)
      logits = logits.detach().cpu().numpy()
      label_ids = labels.to("cpu").numpy()
      pred_flat, labels_flat = get_flats(logits, label_ids)
      pred_flat_sum += pred_flat
      labels_flat_sum += labels_flat


      val_loss += loss

    val_loss = val_loss/len(dataloader)
    return val_loss, pred_flat_sum, labels_flat_sum

def run(model, model_name, optimizer, train_dataloader, valid_dataloader, test_dataloader, epochs=5, best_loss = 10000):
  for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, pred_flat_sum, labels_flat_sum = evaluation(model, valid_dataloader)
    print(f"{model_name} Epoch {epoch + 1}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    print(f"{model_name} labels_flat_sum: {labels_flat_sum}")
    print(f"{model_name} pred_flat_sum: {pred_flat_sum}")
    print(classification_report(labels_flat_sum, pred_flat_sum, labels=[0, 1]))

    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), f"./saved_states/{model_name}_hate_speech_kor_unsmile.pt")
      print(f"Saved {model_name} model weights")

  model.config.pad_token_id = model.config.eos_token_id
  model.load_state_dict(torch.load(f"./saved_states/{model_name}_hate_speech_kor_unsmile.pt"))

  test_loss, pred_flat_sum, labels_flat_sum = evaluation(model, test_dataloader)
  print(f"{model_name} Test Loss : {test_loss:.4f}")
  print(classification_report(labels_flat_sum, pred_flat_sum, labels=[0, 1]))