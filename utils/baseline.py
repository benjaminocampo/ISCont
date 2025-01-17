import torch
import numpy as np
import json
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from huggingface_hub import HfApi, HfFolder

from utils.helpers import multi_label_metrics, predicted_labels_from_logits


class BaselineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.float)


def fit(
    model,
    train_dataloader,
    dev_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    best_model_path,
    main_metric,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    best_metric = -1
    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_f1": [],
        "dev_accuracy": [],
        "dev_precision": [],
        "dev_recall": [],
    }

    for epoch in range(num_epochs):
        total_cont_loss = 0.0
        total_clf_loss = 0.0
        total_loss = 0.0

        for batch in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            total=len(train_dataloader),
        ):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels.long())
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        dev_metrics, _, _, dev_loss = evaluate(
            model=model,
            dataloader=dev_dataloader,
            device=device,
            criterion=criterion,
        )

        print(f"Epoch {epoch + 1} Validation Metrics: {dev_metrics}")

        history["train_loss"].append(avg_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_metrics["f1"])
        history["dev_accuracy"].append(dev_metrics["accuracy"])
        history["dev_precision"].append(dev_metrics["precision"])
        history["dev_recall"].append(dev_metrics["recall"])

        # Save the best model
        if dev_metrics[main_metric] > best_metric:
            best_metric = dev_metrics[main_metric]
            model.push_to_hub(best_model_path)
            print(
                f"Best model of epoch {epoch + 1} saved with F1 score: {best_metric:.4f}"
            )

        history_filename = f"{best_model_path}_history.json"
        with open(history_filename, "w") as json_file:
            json.dump(history, json_file, indent=4)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=history_filename,
            path_in_repo="history.json",
            repo_id=f"BenjaminOcampo/{best_model_path}",
            repo_type="model",
        )
        os.remove(history_filename)


def evaluate(model, dataloader, device, criterion):
    model = model.to(device)
    model.eval()

    total_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            outputs = model(input_ids, attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            loss = criterion(outputs.logits, labels.long())
            total_loss += loss.item()

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(dataloader)

    metrics = multi_label_metrics(logits=all_logits, labels=all_labels)
    y_pred = predicted_labels_from_logits(logits=all_logits, threshold=0.5)
    return metrics, all_labels, y_pred, avg_loss
