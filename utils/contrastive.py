import torch
import numpy as np
import json
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from huggingface_hub import HfApi, HfFolder


class CustomDataset(Dataset):
    def __init__(
        self, texts1, texts2, labels, features, tokenizer, max_length1, max_length2
    ):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer
        self.max_length1 = max_length1
        self.max_length2 = max_length2

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        labels = self.labels[idx]
        features = self.features[idx]

        inputs1 = self.tokenizer(
            text1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length1,
            return_tensors="pt",
        )
        inputs2 = self.tokenizer(
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length2,
            return_tensors="pt",
        )

        input_ids1 = inputs1["input_ids"].squeeze()
        attention_mask1 = inputs1["attention_mask"].squeeze()
        input_ids2 = inputs2["input_ids"].squeeze()
        attention_mask2 = inputs2["attention_mask"].squeeze()

        return (
            input_ids1,
            attention_mask1,
            input_ids2,
            attention_mask2,
            torch.tensor(features, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float),
        )


def fit_constrastive(
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
    cont_criterion = ContrastiveLoss()

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
            (
                input_ids1,
                attention_mask1,
                input_ids2,
                attention_mask2,
                labels_cont,
                labels_clf,
            ) = [item.to(device) for item in batch]

        optimizer.zero_grad()

        embeddings1, logits1 = contrastive_model(input_ids1, attention_mask1)
        embeddings2, logits2 = contrastive_model(input_ids2, attention_mask2)

        cont_loss = cont_criterion(embeddings1, embeddings2, labels_cont)
        clf_loss = criterion(logits1, labels_clf[:, 0]) + criterion(
            logits2, labels_clf[:, 1]
        )
        loss = cont_loss + clf_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_cont_loss += cont_loss.item()
        total_clf_loss += clf_loss.item()
        total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        avg_cont_loss = total_cont_loss / len(data_loader)
        avg_clf_loss = total_clf_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Cont Loss: {avg_cont_loss:.4f}, CLF Loss: {avg_clf_loss:.4f}"
        )

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
