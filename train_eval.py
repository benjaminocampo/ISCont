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
)

import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow
import torch


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


class ClassificationDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        message = self.texts[idx]

        encoding = self.tokenizer(message, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return input_ids, attention_mask


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

        return input_ids1, attention_mask1, input_ids2, attention_mask2, torch.tensor(features, dtype=torch.float), torch.tensor(labels, dtype=torch.float)


def fit(model, x_train, y_train, x_dev, y_dev, num_epochs, optimizer, device):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    cont_criterion = ContrastiveLoss()

    best_metric = -1
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(num_epochs):
        total_cont_loss = 0.0
        total_clf_loss = 0.0
        total_loss = 0.0

        for batch in data_loader:
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

        avg_loss = total_loss / len(data_loader)
        avg_cont_loss = total_cont_loss / len(data_loader)
        avg_clf_loss = total_clf_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Cont Loss: {avg_cont_loss:.4f}, CLF Loss: {avg_clf_loss:.4f}"
        )

        val_metrics, _, _, val_loss = evaluate_model(
            model, val_dataloader, device, criterion
        )

        print(f"Epoch {epoch + 1} Validation Metrics: {val_metrics}")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        # Save the best model
        if val_metrics[args.main_metric] > best_metric:
            best_metric = val_metrics[args.main_metric]
            model.save_pretrained()
            # torch.save(model.state_dict(), best_model_path+'/model.pt')
            print(f"Best model saved with F1 score: {best_metric:.4f}")

        with open(best_model_path + "/history.json", "w") as json_file:
            json.dump(history, json_file, indent=4)


def predicted_labels_from_logits(logits, threshold=0.5):
    """
    Converts logits to binary predictions using a specified threshold.

    Args:
        logits (array-like): The raw output from the model before applying any activation function.
        threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
        np.ndarray: Binary predictions based on the specified threshold.
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred


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
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            loss = criterion(outputs, labels)
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
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        # Instance the model class. NOTE: MODELS is a dictionary where the values
        # are the model classes and the keys their corresponding class names.
        # cfg.model_name matches a key of MODELS.
        # model = cfg.train.model.module(output_dir=output_dir,
        #                               **cfg.train.model.params)

        model_name = f"{cfg.model.params.model_name_or_path.replace('/', '_')}_finetuned_{cfg.model.params.num_epochs}_epochs"
        model_dir = Path(cfg.input.model_dir) / model_name

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}"
        )

        # Load train and dev datasets from `cfg.input.train_file` and `cfg.input.dev_file`.
        train = pd.read_parquet(cfg.input.train_file)
        val = pd.read_parquet(cfg.input.val_file)
        test = pd.read_parquet(cfg.input.test_file)

        # Separate in messages and labels.
        x_train, y_train = train["text"], train["label"].astype(int)
        x_val, y_val = val["text"], val["label"].astype(int)
        x_test, y_test = test["text"], test["label"].astype(int)

        # Shuffle train and val sets.
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        x_val, y_val = shuffle(x_val, y_val, random_state=0)

        # Use only a proportion of the train set to train the model
        if cfg.input.train_size is not None:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train, train_size=cfg.input.train_size
            )

        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ContTransformer(num_labels=cfg.input.num_labels)
        optimizer = AdamW(model.parameters(), lr=cfg.model.params.learning_rate)
        total_steps = len(train_dataloader) * cfg.model.params.num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_step=total_steps
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.params.model_name_or_path)

        train_dataset = CustomDataset(
            texts=x_train, labels=y_train, tokenizer=tokenizer, max_length=128
        )

        val_dataset = CustomDataset(
            texts=x_val, labels=y_val, tokenizer=tokenizer, max_length=128
        )

        test_dataset = CustomDataset(
            texts=x_test, labels=y_test, tokenizer=tokenizer, max_length=128
        )

        if cfg.do_train:
            logger.info("training model...")

            start_time = time.time()
            fit(model=model, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev)
            elapsed_time = time.time() - start_time
            print("Train time:", elapsed_time / 60, "minutes")

            logger.info("saving model...")
            # Save resultant model in `save_dir`.
            # TODO: Add URL.
            model.save_pretrained()

            logger.info("pushing to hub...")
            # TODO: Add URL.
            model.push_to_hub()

            logger.info("training finished succesfully.")
            # Log model to mlflow.
            mlflow.log_artifact(output_dir)

        if cfg.do_eval:
            # TODO: Add URL.
            model = cfg.train.model.module.from_pretrained()

            val_metrics, val_labels, val_predictions, _ = evaluate(
                model=model, x_test=x_test, y_test=y_test
            )
            print(val_metrics)

            test_metrics, test_labels, test_predictions, _ = evaluate(
                model=model, x_test=x_test, y_test=y_test
            )
            print(test_metrics)

            # Make predictions.
            # y_pred_dev = model.predict(x_dev)
            # y_pred_test = model.predict(x_test)

            # Calculate and log metrics.
            report = (
                f"**Classification results dev set**\n```\n{classification_report(y_pred=val_predictions, y_true=val_labels, zero_division='warn')}```\n"
                + f"**Classification results test set**\n```\n{classification_report(y_pred=test_predictions, y_true=test_labels, zero_division='warn')}```\n"
            )
            mlflow.set_tag("mlflow.note.content", report)

            mlflow.log_metrics(val_metrics)
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
