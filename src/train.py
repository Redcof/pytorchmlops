import random

import numpy as np
from dotenv import load_dotenv

from pl_module import CNN_pl_module

load_dotenv()  # make sure python working directory to project root

import lightning
import mlflow.pytorch
import torch
from torch.nn.functional import cross_entropy
from mlflow import MlflowClient

from dataset import MyDataModule
import logging
from ignite.utils import manual_seed

logger = logging.getLogger(__file__)
# create a formatter object
Log_Format = "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
formatter = logging.Formatter(fmt=Log_Format)

# Add custom handler with format to this logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


if __name__ == '__main__':
    # setlogger
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    from lightning import _logger as lightning_console_logger

    lightning_console_logger.setLevel(log_level)
    # define hyperparameters
    IMG_SIZE = 224
    NUM_LABELS = 6
    BATCH_SIZE = 16
    RANDOM_SEED = 37
    DATALOADER_WORKERS = 4
    CRITERION = cross_entropy
    OPTIMIZER = torch.optim.AdamW
    LR = 0.02

    # set random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    manual_seed(RANDOM_SEED)

    # Load dataset.
    datamodule = MyDataModule(im_size=(IMG_SIZE, IMG_SIZE),
                              ratios=(0.65, 0.2, 0.15),
                              batch_size=BATCH_SIZE,
                              workers=DATALOADER_WORKERS)

    # Initialize our model.
    cnn_model = CNN_pl_module(NUM_LABELS, CRITERION, OPTIMIZER, LR,
                              ("matiz", "rio", "tiggo", "black", "blue", "red"))

    # Initialize a trainer.
    # checkpoint_callback = ModelCheckpoint(save_top_k=0)
    trainer = lightning.Trainer(max_epochs=1, accelerator='gpu',
                                callbacks=[],
                                num_sanity_val_steps=1)

    mlflow.set_experiment("pytorch_mlflow")
    mlflow.pytorch.autolog(log_models=False, silent=False,
                           checkpoint=False, checkpoint_monitor="val_f1_score")

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(cnn_model, datamodule)
        trainer.test(cnn_model, datamodule)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
