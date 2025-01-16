from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow
import torch
import time
import random
import numpy as np
import json
import os
from tempfile import NamedTemporaryFile
from torch import nn
from typing import MutableMapping, Dict, Any
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder


load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


torch.cuda.empty_cache()

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
    # exit since there is not GPU support
    # sys.exit(1)

HF_TOKEN = os.environ.get("HF_TOKEN", "")


class ClassificationDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        message = self.texts[idx]

        encoding = self.tokenizer(
            message,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return input_ids, attention_mask


class CustomDataset(Dataset):
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


# class CustomDataset(Dataset):
#    def __init__(
#        self, texts1, texts2, labels, features, tokenizer, max_length1, max_length2
#    ):
#        self.texts1 = texts1
#        self.texts2 = texts2
#        self.labels = labels
#        self.features = features
#        self.tokenizer = tokenizer
#        self.max_length1 = max_length1
#        self.max_length2 = max_length2
#
#    def __len__(self):
#        return len(self.texts1)
#
#    def __getitem__(self, idx):
#        text1 = self.texts1[idx]
#        text2 = self.texts2[idx]
#        labels = self.labels[idx]
#        features = self.features[idx]
#
#        inputs1 = self.tokenizer(
#            text1,
#            truncation=True,
#            padding="max_length",
#            max_length=self.max_length1,
#            return_tensors="pt",
#        )
#        inputs2 = self.tokenizer(
#            text2,
#            truncation=True,
#            padding="max_length",
#            max_length=self.max_length2,
#            return_tensors="pt",
#        )
#
#        input_ids1 = inputs1["input_ids"].squeeze()
#        attention_mask1 = inputs1["attention_mask"].squeeze()
#        input_ids2 = inputs2["input_ids"].squeeze()
#        attention_mask2 = inputs2["attention_mask"].squeeze()
#
#        return (
#            input_ids1,
#            attention_mask1,
#            input_ids2,
#            attention_mask2,
#            torch.tensor(features, dtype=torch.float),
#            torch.tensor(labels, dtype=torch.float),
#        )


def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep).items()
        elif isinstance(v, list) or isinstance(v, list):
            #  For lists we transform them into strings with a join
            yield new_key, "#".join(map(str, v))
        else:
            yield new_key, v


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a dictionary using recursion (via an auxiliary funciton).
    The list/tuples values are flattened as a string.

    Parameters
    ----------
    d : MutableMapping
        Dictionary (or, more generally something that is a MutableMapping) to flatten.
        It might be nested, thus the function will traverse it to flatten it.
    parent_key : str
        Key of the parent dictionary in order to append to the path of keys.
    sep : str
        Separator to use in order to represent nested structures.

    Returns
    -------
    Dict[str, Any]
        The flattened dict where each nested dictionary is expressed as a path with
        the `sep`.

    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}})
    {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
    >>> flatten_dict({'a': {'b': [1, 2]}})
    {'a.b': '1#2'}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))


# def fit(model, x_train, y_train, x_dev, y_dev, num_epochs, optimizer, device):


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
    # cont_criterion = ContrastiveLoss()

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
            # (
            #    input_ids1,
            #    attention_mask1,
            #    input_ids2,
            #    attention_mask2,
            #    labels_cont,
            #    labels_clf,
            # ) = [item.to(device) for item in batch]
        #
        # optimizer.zero_grad()
        #
        # embeddings1, logits1 = contrastive_model(input_ids1, attention_mask1)
        # embeddings2, logits2 = contrastive_model(input_ids2, attention_mask2)
        #
        # cont_loss = cont_criterion(embeddings1, embeddings2, labels_cont)
        # clf_loss = criterion(logits1, labels_clf[:, 0]) + criterion(
        #    logits2, labels_clf[:, 1]
        # )
        # loss = cont_loss + clf_loss
        #
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        #
        # total_cont_loss += cont_loss.item()
        # total_clf_loss += clf_loss.item()
        # total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        # avg_cont_loss = total_cont_loss / len(data_loader)
        # avg_clf_loss = total_clf_loss / len(data_loader)

        # print(
        #    f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Cont Loss: {avg_cont_loss:.4f}, CLF Loss: {avg_clf_loss:.4f}"
        # )

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
            # torch.save(model.state_dict(), best_model_path+'/model.pt')
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


def predicted_labels_from_logits(logits, threshold=0.5):
    """
    Converts logits to binary predictions using a specified threshold.

    Args:
        logits (array-like): The raw output from the model before applying any activation function.
        threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
        np.ndarray: Binary predictions based on the specified threshold.
    """
    # sigmoid = torch.nn.Sigmoid()
    # probs = sigmoid(torch.Tensor(logits))
    # y_pred = np.zeros(probs.shape)
    # y_pred[np.where(probs >= threshold)] = 1
    probs = torch.softmax(torch.FloatTensor(logits), dim=1)
    pred_labels = torch.max(probs, dim=1)
    return pred_labels.indices


def multi_label_metrics(logits, labels, threshold=0.5):
    y_pred = predicted_labels_from_logits(logits=logits, threshold=threshold)
    y_true = labels

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}
    return metrics


def concatenate_columns(row, columns):
    """
    Concatenate the values of specified columns in a pandas DataFrame row into a single string,
    treating NaN values as empty strings.

    Args:
        row (pandas.Series): A single row from a pandas DataFrame.
        columns (list of str): A list of column names whose values will be concatenated.

    Returns:
        str: A single string containing the concatenated values from the specified columns of the row.
    """
    concatenated = ""
    for column in columns:
        concatenated += f"{row[column] if pd.notna(row[column]) else ''}"
    return concatenated


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


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Script that finetunes/train a model.
    """
    set_seed(cfg.model.params.seed)
    login(token=HF_TOKEN)

    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        # Instance the model class. NOTE: MODELS is a dictionary where the values
        # are the model classes and the keys their corresponding class names.
        # cfg.model_name matches a key of MODELS.
        # model = cfg.train.model.module(output_dir=output_dir,
        #                               **cfg.train.model.params)

        model_name = (
            f"{cfg.model.params.model_name_or_path.replace('/', '_')}_finetuned"
        )
        # model_dir = Path(cfg.input.model_dir) / model_name

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        # Load train and dev datasets from `cfg.input.train_file` and `cfg.input.dev_file`.
        train = pd.read_parquet(cfg.input.train_file)
        dev = pd.read_parquet(cfg.input.dev_file)
        test = pd.read_parquet(cfg.input.test_file)

        train = train.sample(100, ignore_index=True)
        dev = dev.sample(100, ignore_index=True)
        test = test.sample(100, ignore_index=True)

        # Separate in messages and labels.
        # import pdb; pdb.set_trace()
        x_train, y_train = train["text"], train["label"].replace({"hs": 1, "non-hs": 0})
        x_dev, y_dev = dev["text"], dev["label"].replace({"hs": 1, "non-hs": 0})
        x_test, y_test = test["text"], test["label"].replace({"hs": 1, "non-hs": 0})

        # Shuffle train and dev sets.
        # x_train, y_train = shuffle(x_train, y_train, random_state=0)
        # x_dev, y_dev = shuffle(x_dev, y_dev, random_state=0)

        # Use only a proportion of the train set to train the model
        if cfg.input.train_size is not None:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train, train_size=cfg.input.train_size
            )

        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = ContTransformer(num_labels=cfg.input.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased", num_labels=cfg.model.params.num_labels
        )
        optimizer = AdamW(model.parameters(), lr=cfg.model.params.learning_rate)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.params.model_name_or_path)

        train_dataset = CustomDataset(
            texts=x_train, labels=y_train, tokenizer=tokenizer, max_length=128
        )

        dev_dataset = CustomDataset(
            texts=x_dev, labels=y_dev, tokenizer=tokenizer, max_length=128
        )

        test_dataset = CustomDataset(
            texts=x_test, labels=y_test, tokenizer=tokenizer, max_length=128
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=cfg.model.params.batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=cfg.model.params.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=cfg.model.params.batch_size, shuffle=False
        )

        total_steps = len(train_dataloader) * cfg.model.params.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        if cfg.input.do_train:
            logger.info("training model...")

            start_time = time.time()
            # model, train_dataloader, dev_dataloader, optimizer, scheduler, device, num_epochs, best_model_path

            fit(
                model=model,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=cfg.model.params.num_epochs,
                best_model_path=model_name,
                main_metric=cfg.model.params.main_metric,
            )
            elapsed_time = time.time() - start_time
            print("Train time:", elapsed_time / 60, "minutes")

            # logger.info("saving model...")
            # Save resultant model in `save_dir`.
            # TODO: Add URL.
            # model.save_pretrained(model_name)

            # Authenticate
            # token = HfFolder.get_token()
            # import pdb; pdb.set_trace()
            # api = HfApi()
            # api.create_repo(repo_id=model_name, token=HF_TOKEN)

            # logger.info("pushing to hub...")
            ## TODO: Add URL.
            # model.push_to_hub(repo_id=model_name, token=HF_TOKEN)

            logger.info("training finished succesfully.")
            ## Log model to mlflow.
            # mlflow.log_artifact(output_dir)

        if cfg.input.do_eval:
            # TODO: Add URL.
            model = AutoModelForSequenceClassification.from_pretrained(
                f"BenjaminOcampo/{model_name}"
            )

            dev_metrics, dev_labels, dev_predictions, _ = evaluate(
                model, dev_dataloader, device, criterion=nn.CrossEntropyLoss()
            )
            print(dev_metrics)

            test_metrics, test_labels, test_predictions, _ = evaluate(
                model, test_dataloader, device, criterion=nn.CrossEntropyLoss()
            )
            print(test_metrics)

            # Calculate and log metrics.
            report = (
                f"**Classification results dev set**\n```\n{classification_report(y_pred=dev_predictions, y_true=dev_labels, zero_division='warn')}```\n"
                + f"**Classification results test set**\n```\n{classification_report(y_pred=test_predictions, y_true=test_labels, zero_division='warn')}```\n"
            )
            mlflow.set_tag("mlflow.note.content", report)

            mlflow.log_metrics(dev_metrics)
            mlflow.log_metrics(test_metrics)

    print("---------------------------------")
    print("Done!----------------------------")
    print("---------------------------------")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    if cfg.input.uri_path is not None:
        mlflow.set_tracking_uri(cfg.input.uri_path)
        assert cfg.input.uri_path == mlflow.get_tracking_uri()

    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    mlflow.set_experiment(cfg.input.experiment_name)
    mlflow.set_experiment_tag("mlflow.note.content", cfg.input.experiment_description)

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Log params expects a flatten dictionary, since the configuration has nested
        # configurations (e.g. train.model), we need to use flatten_dict in order to
        # transform it into something that can be easilty logged by MLFlow.
        mlflow.log_params(flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run)


if __name__ == "__main__":
    main()
