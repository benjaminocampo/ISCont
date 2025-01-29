import torch
import numpy as np
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfApi, HfFolder, PyTorchModelHubMixin
from transformers import AutoModel
from utils.helpers import multi_label_metrics, predicted_labels_from_logits


class ContrastiveModel(
    nn.Module, PyTorchModelHubMixin, pipeline_tag="text-classification"
):
    def __init__(self, model_name_or_path, num_labels):
        super(ContrastiveModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.num_labels = num_labels
        self.embedding_dim = self.model.config.hidden_size
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.classifier = nn.Linear(
            self.embedding_dim, self.num_labels
        )  # Classification layer

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)

        # Some models do not provide a pooler_output, use last_hidden_state if that is the case
        if hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            embeddings = outputs.pooler_output

        # We can use the CLS token as well with
        # embeddings = outputs.last_hidden_state[ :, 0 ]

        embeddings = self.fc(embeddings)
        logits = self.classifier(embeddings)  # Apply classification layer

        return embeddings, logits


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, embeddings1, embeddings2, target):
        similarity = self.cosine_similarity(embeddings1, embeddings2)
        loss = torch.mean(
            (1 - target) * torch.pow(similarity, 2)
            + target * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)
        )
        return loss


class ContrastiveDataset(Dataset):

    def __init__(self, texts, labels, labels_cont, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.labels_cont = labels_cont
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text1, text2 = self.texts[idx]
        labels = self.labels[idx]
        labels_cont = self.labels_cont[idx]

        inputs1 = self.tokenizer(
            text1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs2 = self.tokenizer(
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
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
            torch.tensor(labels_cont, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float),
        )


def fit_contrastive(
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
        "dev_cont_loss": [],
        "dev_clf_loss": [],
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

            logits1, embeddings1 = model(input_ids1, attention_mask1)
            logits2, embeddings2 = model(input_ids2, attention_mask2)

            cont_loss = cont_criterion(embeddings1, embeddings2, labels_cont)
            clf_loss = criterion(logits1, labels_clf[:, 0].long()) + criterion(
                logits2, labels_clf[:, 1].long()
            )
            loss = cont_loss + clf_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_cont_loss += cont_loss.item()
            total_clf_loss += clf_loss.item()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        avg_cont_loss = total_cont_loss / len(train_dataloader)
        avg_clf_loss = total_clf_loss / len(train_dataloader)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Cont Loss: {avg_cont_loss:.4f}, CLF Loss: {avg_clf_loss:.4f}"
        )

        dev_metrics, _, _, dev_loss, dev_cont_loss, dev_clf_loss = evaluate_contrastive(
            model=model,
            dataloader=dev_dataloader,
            device=device,
            criterion=criterion,
        )

        print(f"Epoch {epoch + 1} Validation Metrics: {dev_metrics}")

        history["train_loss"].append(avg_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_cont_loss"].append(dev_loss)
        history["dev_clf_loss"].append(dev_loss)
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


def evaluate_contrastive(model, dataloader, device, criterion):
    model = model.to(device)
    model.eval()
    cont_criterion = ContrastiveLoss()

    total_loss = 0
    total_cont_loss = 0
    total_clf_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            (
                input_ids1,
                attention_mask1,
                input_ids2,
                attention_mask2,
                labels_cont,
                labels_clf,
            ) = [item.to(device) for item in batch]

            logits1, embeddings1 = model(input_ids1, attention_mask1)
            logits2, embeddings2 = model(input_ids2, attention_mask2)

            cont_loss = cont_criterion(embeddings1, embeddings2, labels_cont)
            clf_loss = criterion(logits1, labels_clf[:, 0].long()) + criterion(
                logits2, labels_clf[:, 1].long()
            )
            loss = cont_loss + clf_loss
            total_cont_loss += cont_loss.item()
            total_clf_loss += clf_loss.item()
            total_loss += loss.item()

            all_logits.append(
                np.concatenate([logits1.cpu().numpy(), logits2.cpu().numpy()], axis=0)
            )
            all_labels.append(
                np.concatenate(
                    [labels_clf[:, 0].cpu().numpy(), labels_clf[:, 1].cpu().numpy()]
                )
            )

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(dataloader)
    avg_cont_loss = total_cont_loss / len(dataloader)
    avg_clf_loss = total_clf_loss / len(dataloader)

    metrics = multi_label_metrics(logits=all_logits, labels=all_labels)
    y_pred = predicted_labels_from_logits(logits=all_logits, threshold=0.5)
    return metrics, all_labels, y_pred, avg_loss, avg_cont_loss, avg_clf_loss


def create_pos_pairs(data, threshold, drop_ids_and_IS=True):
    imp_df = data[data["implicitness"] == "yes"]
    exp_df = data[data["implicitness"] == "no"]

    pos_pairs = []
    for row_id, imp_row in tqdm(
        imp_df.iterrows(),
        desc="Creating positive pairs",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        total=len(imp_df),
    ):
        imp_text = imp_row["text"]
        imp_IS = imp_row["IS_prep"]
        imp_target = imp_row["sanitized_target"]
        # pairs = []
        exp_same_target = exp_df[exp_df["sanitized_target"] == imp_target]
        for _, exp_row in exp_same_target.iterrows():
            exp_text = exp_row["text"]

            imp_IS_emb = imp_row["IS_emb"]
            exp_emb = exp_row["text_emb"]

            cos_sim = euclidean_distances([imp_IS_emb], [exp_emb])
            if (1 - cos_sim) < threshold:
                pos_pairs.append((row_id, imp_text, exp_text, imp_IS))

        # pos_pairs.append(pairs)
    if drop_ids_and_IS:
        pos_pairs = [(t[1], t[2]) for t in pos_pairs]

    return pos_pairs


def create_neg_pairs(data):
    imp_hs_df = data[(data["label"] == 1) & (data["implicitness"] == "yes")]
    non_hs_df = data[data["label"] == 0]

    neg_pairs = []

    for _, hs_row in tqdm(
        imp_hs_df.iterrows(),
        desc="Creating negative pairs",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        total=len(imp_hs_df),
    ):
        imp_hs_message = hs_row["text"]
        imp_hs_target = hs_row["sanitized_target"]
        non_hs_same_target = non_hs_df[non_hs_df["sanitized_target"] == imp_hs_target]
        for _, non_hs_row in non_hs_same_target.iterrows():
            non_hs_message = non_hs_row["text"]

            if (non_hs_message is not None) and (imp_hs_message is not None):
                # Append negative pair to the list
                neg_pairs.append((imp_hs_message, non_hs_message))

    return neg_pairs

import random

def prepare_cont_data(
    data, tokenizer, max_length, batch_size, do_shuffle, threshold=1e-1
):
    pos_pairs = create_pos_pairs(data=data, threshold=threshold)
    neg_pairs = create_neg_pairs(data=data)

    print(len(pos_pairs))
    print(len(neg_pairs))
    pos_pairs = random.sample(pos_pairs, 500)
    neg_pairs = random.sample(neg_pairs, 500)
    all_pairs = pos_pairs + neg_pairs

    labels_cont = torch.tensor([1] * len(pos_pairs) + [0] * len(neg_pairs))
    labels_clf = torch.tensor(
        [
            [1] * len(pos_pairs) + [1] * len(neg_pairs),
            [1] * len(pos_pairs) + [0] * len(neg_pairs),
        ]
    )
    labels_clf = labels_clf.T

    dataset = ContrastiveDataset(
        texts=all_pairs,
        labels=labels_clf,
        labels_cont=labels_cont,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=do_shuffle)

    return dataloader
