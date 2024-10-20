import os.path
from typing import Union

import numpy as np
import onnx
from dotenv import load_dotenv
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pl_module import CNN_pl_module

load_dotenv()  # make sure python working directory to project root

import lightning
import mlflow.pytorch
import torch
from torch.nn.functional import cross_entropy
from mlflow import MlflowClient

from dataset import MyDataModule
import logging

logger = logging.getLogger("pytorch_mlops")


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


def configure_logger(level: Union[int, str] = logging.INFO) -> None:
    """Get console logger by name.

    Args:
        level (int | str, optional): Logger Level. Defaults to logging.INFO.

    Returns:
        Logger: The expected logger.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string)

    # logger.setLevel(level)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter(format_string))
    # logger.addHandler(console_handler)

    # Set Pytorch Lightning logs to have a consistent formatting with anomalib.
    for handler in logging.getLogger().handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)

    # Set Pytorch Lightning logs to have a consistent formatting with anomalib.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)

    for handler in logging.getLogger("lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)

    for handler in logging.getLogger("mlflow").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)


if __name__ == '__main__':
    # set-logger
    configure_logger(logging.DEBUG)
    # define hyperparameters
    IMG_SIZE = 224
    NUM_LABELS = 6
    BATCH_SIZE = 16
    RANDOM_SEED = 37
    DATALOADER_WORKERS = 4
    CRITERION = cross_entropy
    OPTIMIZER = torch.optim.AdamW
    performance_metric = "val_f1_score"
    MAX_EPOCH = 1
    LR = 0.02
    input_sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # set random seed
    seed_everything(RANDOM_SEED)

    # Load dataset.
    datamodule = MyDataModule(im_size=(IMG_SIZE, IMG_SIZE),
                              ratios=(0.65, 0.2, 0.15),
                              batch_size=BATCH_SIZE,
                              workers=DATALOADER_WORKERS)

    # Initialize our model.
    cnn_model = CNN_pl_module(NUM_LABELS, CRITERION, OPTIMIZER, LR,
                              ("matiz", "rio", "tiggo", "black", "blue", "red"))

    # callbacks
    model_checkpoint_callback = ModelCheckpoint(verbose=True, monitor=performance_metric,
                                                save_weights_only=True,
                                                filename='{epoch}-{%s}' % performance_metric)
    callbacks = [
        model_checkpoint_callback,
        EarlyStopping(monitor=performance_metric, min_delta=0.001,
                      patience=5, verbose=True,
                      check_on_train_epoch_end=True),
    ]

    # Initialize a trainer.
    trainer = lightning.Trainer(max_epochs=MAX_EPOCH,
                                accelerator='gpu',
                                callbacks=callbacks,
                                num_sanity_val_steps=1)

    mlflow.set_experiment("pytorch_mlflow")
    mlflow.pytorch.autolog(
        # model logging setting
        log_models=True,
        # model monitoring settings
        checkpoint=False,
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=True,
        checkpoint_monitor=performance_metric,
        silent=False,
    )

    # Train the model.
    with mlflow.start_run() as run:
        # train model
        logger.info("Training...")
        trainer.fit(model=cnn_model, datamodule=datamodule)
        # load best model
        logger.info("Loading checkpoint...")
        cnn_model = CNN_pl_module.load_from_checkpoint(model_checkpoint_callback.best_model_path,
                                                       num_labels=NUM_LABELS,
                                                       criterion=CRITERION,
                                                       optimizer=cnn_model.optimizer,
                                                       lr=LR,
                                                       labels=["matiz", "rio", "tiggo", "black", "blue", "red"]
                                                       )
        # test
        logger.info("Testing model...")
        trainer.test(model=cnn_model, datamodule=datamodule)
        # export onnx
        filename = os.path.basename(model_checkpoint_callback.best_model_path)
        onnx_filename = filename.replace(".ckpt", ".onnx")
        logger.info("Saving model to onnx format...")
        cnn_model.to_onnx(onnx_filename, input_sample, export_params=True)
        # save to mlflow
        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)
        logger.info("Uploading onnx-model to mlflow...")
        mlflow.onnx.log_model(onnx_model,
                              "onnx",
                              input_example=np.zeros((1, 3, IMG_SIZE, IMG_SIZE)),
                              conda_env=mlflow.onnx.get_default_conda_env())
        logger.info("All done")
