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
from torch.utils.data import Dataset

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
from torch import nn
from tqdm import tqdm
from huggingface_hub import login, HfApi, HfFolder
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification

from utils.baseline import BaselineDataset, fit, evaluate
from utils.helpers import flatten_dict, set_seed
from utils.contrastive import fit_constrastive

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


def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Script that finetunes/train a model.
    """
    set_seed(cfg.model.params.seed)
    login(token=HF_TOKEN)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"{cfg.train.model.params.model_name_or_path.replace('/', '_')}_seed-{cfg.model.params.seed}_finetuned"

    logger.info("Command-line Arguments:")
    logger.info(f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}")

    train = pd.read_parquet(cfg.input.train_file)
    dev = pd.read_parquet(cfg.input.dev_file)
    test = pd.read_parquet(cfg.input.test_file)

    train = train.sample(100, ignore_index=True)
    dev = dev.sample(100, ignore_index=True)
    test = test.sample(100, ignore_index=True)

    x_train, y_train = train["text"], train["label"].replace({"hs": 1, "non-hs": 0})
    x_dev, y_dev = dev["text"], dev["label"].replace({"hs": 1, "non-hs": 0})
    x_test, y_test = test["text"], test["label"].replace({"hs": 1, "non-hs": 0})

    if cfg.input.train_size is not None:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=cfg.input.train_size
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.train.model.params.model_name_or_path)

    train_dataset = cfg.train.model.data_module(
        texts=x_train, labels=y_train, tokenizer=tokenizer, max_length=128
    )

    dev_dataset = cfg.train.model.data_module(
        texts=x_dev, labels=y_dev, tokenizer=tokenizer, max_length=128
    )

    test_dataset = cfg.train.model.data_module(
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

    model = cfg.train.model.module(
        cfg.train.model.params.model_name_or_path, num_labels=cfg.model.params.num_labels
    )
    optimizer = AdamW(model.parameters(), lr=cfg.model.params.learning_rate)
    total_steps = len(train_dataloader) * cfg.model.params.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    if cfg.input.do_train:
        logger.info("training model...")

        start_time = time.time()

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

        logger.info("training finished succesfully.")

    if cfg.input.do_eval:
        logger.info("evaluating model...")

        start_time = time.time()
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

        elapsed_time = time.time() - start_time
        print("Evaluation time:", elapsed_time / 60, "minutes")
        logger.info("Evaluation finished succesfully.")

        report = (
            f"**Classification results dev set**\n```\n{classification_report(y_pred=dev_predictions, y_true=dev_labels, zero_division='warn')}```\n"
            + f"**Classification results test set**\n```\n{classification_report(y_pred=test_predictions, y_true=test_labels, zero_division='warn')}```\n"
        )
        mlflow.set_tag("mlflow.note.content", report)

        mlflow.log_metrics(dev_metrics)
        mlflow.log_metrics(test_metrics)

    logger.info("Experiment finished succesfully.")


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
