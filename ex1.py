import argparse
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import sys
import os

def encode(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

def train(args, model, tokenizer, train_loader, val_loader, device):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"train_loss": loss.item()})

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")
        wandb.log({"val_accuracy": val_acc})

    # Save final model
    save_path = f"lr_{args.lr}_epoch_{args.num_train_epochs}_bs_{args.batch_size}_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Append result to res.txt
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {best_val_acc:.4f}\n")

    return best_val_acc

def predict(args, model, tokenizer, data_loader, dataset_split, device, split_name="test"):
    model.eval()
    predictions = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            sent1 = dataset_split[i]['sentence1']
            sent2 = dataset_split[i]['sentence2']
            line = f"{sent1}###{sent2}###{pred}"
            predictions.append(line)
            pred_labels.append(pred)

    # Save to file
    if split_name == "test":
        with open("predictions.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(predictions))
    else:
        pred_filename = f"{split_name}_predictions_{args.num_train_epochs}_{args.lr}_{args.batch_size}.txt"
        with open(pred_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(predictions))
    print(f"Predictions saved for {split_name} split.")
    return pred_labels


def compute_test_accuracy(pred_labels, true_labels):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    acc = np.mean(pred_labels == true_labels)
    print(f"Test Accuracy: {acc:.4f}")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    # === Automatically create model path from args if not provided ===
    generated_model_path = f"lr_{args.lr}_epoch_{args.num_train_epochs}_bs_{args.batch_size}_model"
    if not args.model_path:
        args.model_path = generated_model_path

    run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
    wandb.init(project="bert-mrpc-paraphrase", name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    encoded_dataset = dataset.map(lambda x: encode(x, tokenizer), batched=True)
    encoded_dataset = encoded_dataset.map(lambda x: {'labels': x['label']}, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_data = encoded_dataset['train']
    val_data = encoded_dataset['validation']
    test_data = encoded_dataset['test']

    if args.max_train_samples != -1:
        train_data = train_data.select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        val_data = val_data.select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        test_data = test_data.select(range(args.max_predict_samples))

    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, collate_fn=data_collator)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, collate_fn=data_collator)

    if args.do_train:
        val_acc = train(args, model, tokenizer, train_loader, val_loader, device)
        print(f"Final Validation Accuracy: {val_acc:.4f}")

    if args.do_predict:
      if not os.path.exists(args.model_path):
          print(f"ERROR: model path '{args.model_path}' does not exist. Did you forget to run with --do_train?")
          sys.exit(1)

      model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2).to(device)

      # Predict on test set
      test_pred_labels = predict(args, model, tokenizer, test_loader, dataset['test'], device, split_name="test")
      true_test_labels = np.array(dataset['test']['label'])
      compute_test_accuracy(test_pred_labels, true_test_labels) 


# Automatically run with default config inside Colab
if __name__ == "__main__":
    sys.argv = [
        "ex1.py", "--do_train", "--do_predict",
        "--num_train_epochs", "2", "--lr", "5e-5", "--batch_size", "16"
    ]
    main()
