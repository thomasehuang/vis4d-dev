# pylint: disable=duplicate-code
"""QDTrack with YOLOX-x on BDD100K."""
from __future__ import annotations

import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.datasets.bdd100k import bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import (
    EvaluatorCallback,
    FreezeCallback,
    VisualizerCallback,
)
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.zoo.base.datasets.bdd100k import CONN_BDD100K_TRACK_EVAL
from vis4d.zoo.base.models.qdtrack import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    get_qdtrack_yolox_cfg,
)
from vis4d.zoo.qdtrack.data_yolox import get_bdd100k_track_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_yolox_x_frz_1x_bdd100k_ft")
    config.checkpoint_period = 2
    config.check_val_every_n_epoch = 2

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 8  # batch size = 8 GPUs * 8 samples per GPU = 64
    params.workers_per_gpu = 8
    params.lr = 0.08
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_track_cfg(
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)
    weights = (
        "ema://vis4d-workspace/yolox_x_25e_bdd100k25/train/"
        "checkpoints/last.ckpt"
    )
    config.model, config.loss = get_qdtrack_yolox_cfg(
        num_classes,
        "xlarge",
        use_ema=False,
        weights=weights,
        only_track_loss=True,
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(LinearLR, start_factor=0.1, total_iters=1000),
                    end=1000,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(
        config.output_dir, refresh_rate=config.log_every_n_steps
    )

    # Freeze model parameters
    callbacks.append(
        class_config(
            FreezeCallback,
            freeze_keys=["basemodel", "fpn", "yolox_head"],
            verbose=True,
        )
    )

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBoxVisualizer, vis_freq=500, image_mode="BGR"
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_TRACK_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KTrackEvaluator,
                annotation_path="data/bdd100k/labels/box_track_20/val/",
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_TRACK_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.check_val_every_n_epoch = config.check_val_every_n_epoch
    pl_trainer.checkpoint_callback = class_config(
        ModelCheckpoint,
        dirpath=config.get_ref("output_dir") + "/checkpoints",
        verbose=True,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=config.checkpoint_period,
        save_top_k=3,
        mode="max",
        monitor="step",
    )
    pl_trainer.wandb = False
    pl_trainer.inference_mode = False
    pl_trainer.find_unused_parameters = True
    pl_trainer.precision = "16-mixed"
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
