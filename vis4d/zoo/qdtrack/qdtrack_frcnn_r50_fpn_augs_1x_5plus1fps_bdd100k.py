# pylint: disable=duplicate-code
"""QDTrack with Faster R-CNN on BDD100K."""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.common.datasets.bdd100k import CONN_BDD100K_TRACK_EVAL
from vis4d.config.common.models.qdtrack import get_qdtrack_cfg
from vis4d.config.default import get_default_callbacks_cfg
from vis4d.config.default.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import (
    EvaluatorCallback,
    VisualizerCallback,
    YOLOXModeSwitchCallback,
)
from vis4d.engine.connectors import CallbackConnector
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.qdtrack.data_yolox_5plus1fps import get_bdd100k_track_cfg
from vis4d.zoo.qdtrack.qdtrack_frcnn_r50_fpn_augs_1x_bdd100k import (
    get_config as get_qdtrack_cfg,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_qdtrack_cfg().ref_mode()

    config.experiment_name = "qdtrack_frcnn_r50_fpn_augs_1x_5plus1fps_bdd100k"

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_track_cfg(
        data_backend=data_backend,
        image_size=(720, 1280),
        normalize_image=True,
        samples_per_gpu=config.params.get().samples_per_gpu,
        workers_per_gpu=config.params.get().workers_per_gpu,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Mode switch for strong augmentations
    callbacks += [class_config(YOLOXModeSwitchCallback, switch_epoch=9)]

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
    config.pl_trainer.get().wandb = False

    return config.value_mode()
