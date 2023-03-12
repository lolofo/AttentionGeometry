from datasets import load_dataset
from transformers import Trainer
from transformers import AutoTokenizer
import torch.cuda as cuda
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
import argparse
import os, sys


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)


if __name__=="__main__":
  parser = argparse.ArgumentParser("Vanilla trainings")
  parser.add_argument("-e", "--epoch", type=int, default=3)
  parser.add_argument("-b", "--batch_size", type=int, default=16)
  parser.add_argument("-n", "--nb_data", type=int, default=20000)
  parser.add_argument("-d", "--data", type=str, default="yelp-polarity")
  parser.add_argument("-o", "--output_dir", type=str, default="./.cache")
  args = parser.parse_args()

  dataset = load_dataset(args.data, split='train')
  train_small = dataset.shuffle(seed=42).select(range(args.nb_data))
  eval_small = dataset.shuffle(seed=42).select(range(args.nb_data,args.nb_data+5000)) # 5000 data for evalutation

  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  device = 'cuda:0' if cuda.is_available() else 'cpu'
  print(device)

  train_data = train_small.map(tokenize_function, batched=True)#.to(device)
  eval_data = eval_small.map(tokenize_function, batched=True)


  model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)



  training_args = TrainingArguments(
      output_dir=args.output_dir,          # output directory
      num_train_epochs=3,              # total number of training epochs
      per_device_train_batch_size=args.batch_size,  # batch size per device during training
      per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      learning_rate = 0.000001,
      logging_dir='./logs',            # directory for storing logs
      logging_steps=1000,
      save_steps=5000,
      evaluation_strategy="epoch",
      save_total_limit = 1, # pour eviter tous les checkpointt
  )



  metric = evaluate.load("accuracy")

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_data,
      eval_dataset=eval_data,
      compute_metrics=compute_metrics,
  )
  print(trainer.args.device)

  print ( f"                              num_labels: {model.num_labels}")
  print ( f"                              classifier: {model.classifier}")


  trainer.train()
  model.save_pretrained(os.path.join(args.output_dir, "models", f"{args.data}-ft-{args.epochs}"))