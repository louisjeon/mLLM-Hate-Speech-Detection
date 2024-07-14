import numpy as np
import pandas as pd

def kor_unsmile():
  df = pd.read_csv('unsmile_train_v1.0.tsv', delimiter="\t")[:1000]
  train, test = np.split(df.sample(frac=1, random_state=42), [int(0.8 * len(df))])
  valid = pd.read_csv('unsmile_valid_v1.0.tsv', delimiter="\t")[:200]

  print(train.head(3).to_markdown())
  print(f"Training Data Size : {len(train)}")
  print(f"Validation Data Size : {len(valid)}")
  print(f"Testing Data Size : {len(test)}")

  return train, test, valid

def eng_curated():
  df = pd.read_csv('curated.csv')[:1000]
  train, valid, test = np.split(df.sample(frac=1, random_state=22), [int(0.6 * len(df)), int(0.8*len(df))])

  print(train.head(3).to_markdown())
  print(f"Training Data Size : {len(train)}")
  print(f"Validation Data Size : {len(valid)}")
  print(f"Testing Data Size : {len(test)}")

  return train, test, valid