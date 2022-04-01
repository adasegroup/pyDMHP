from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
import torch.optim as optim
from torch.optim import lr_scheduler

from src.utils import get_logger
from src.utils.metrics import purity

log = get_logger(__name__)


def train_dmhp(config: DictConfig):
    """
    Training DMHP module for clustering of event sequences
    """
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)
    default_save_dir = config.save_dir
    # Init and prepare lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    dm: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    dm.prepare_data()

    for i in range(config.n_runs):

        config.save_dir = str(Path(default_save_dir, "exp_" + str(i)))
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Run: {i+1}")
        log.info(f"Dataset: {config.data_name}")
        # Init callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "dirpath" in cb_conf:
                    cb_conf.dirpath = config.save_dir
                if "_target_" in cb_conf:
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init Lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

        dm.setup(stage="fit")
        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        config.model.num_type = dm.train_data.num_events
        config.model.num_clusters = dm.train_data.num_clusters
        config.model.num_sequence = len(dm.train_data["seq2idx"])
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Train the model
        log.info("Starting training")
        optimizer = optim.Adam(model.lambda_model.parameters(), lr=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        # train model
        model.fit(
            dm.train_dataloader(),
            optimizer,
            config.num_epochs,
            scheduler=scheduler,
            sparsity=100,
            nonnegative=0,
            use_cuda=config.model.use_cuda,
            validation_set=dm.val_dataloader(),
        )
        # save model
        model.save_model(config.save_dir, mode="entire")
        # model.save_model(config.save_dir, mode="parameter")
        # Inference - cluster labels
        log.info("Starting predicting labels")
        dm.setup(stage="test")
        r = model.responsibility
        clusters_prediction = np.argmax(r.detach().cpu().numpy(), axis=1) # r - responsobility matrix with probabilities
        
        pred_labels = model.final_labels
        gt_labels = dm.target

        clusters_predicted_users = {float(database['idx2seq'][i]):cl for i, cl in enumerate(clusters_prediction)}
        df['clust_pred'] = df['id'].apply(lambda x: clusters_predicted_users[x])


        # Saving predicted and actual labels - for graphs and tables
        df = pd.DataFrame(columns=["cluster_id", "cluster_pred"])
        df["cluster_id"] = gt_labels
        df["cluster_pred"] = pred_labels.tolist()
        df.to_csv(Path(config.save_dir, "inferredclusters.csv"))
