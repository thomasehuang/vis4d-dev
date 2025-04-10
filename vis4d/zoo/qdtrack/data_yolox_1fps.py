"""BDD100K data loading config for QDTrack YOLOX."""

from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe, MultiSampleDataPipe
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.loader import build_train_dataloader, default_collate
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.zoo.base import get_inference_dataloaders_cfg
from vis4d.zoo.base.callable import get_callable_cfg
from vis4d.zoo.qdtrack.data_yolox import (
    get_test_transforms,
    get_train_transforms,
)


def get_train_dataloader(
    data_backend: None | ConfigDict,
    image_size: tuple[int, int],
    normalize_image: bool,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K tracking."""
    bdd100k_det_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/100k/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/det_20/det_train.json",
        category_map=bdd100k_track_map,
        config_path="det",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/det_train.pkl",
    )

    bdd100k_track_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track_1fps.hdf5",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/box_track_1fps/train.json",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_1fps_train.pkl",
    )

    train_dataset_cfg = [
        class_config(
            MultiViewDataset,
            dataset=bdd100k_det_train,
            sampler=class_config(
                UniformViewSampler, scope=0, num_ref_samples=1
            ),
        ),
        class_config(
            MultiViewDataset,
            dataset=bdd100k_track_train,
            sampler=class_config(
                UniformViewSampler, scope=1, num_ref_samples=1
            ),
        ),
    ]

    preprocess_transforms, train_batchprocess_cfg = get_train_transforms(
        image_size=image_size, normalize_image=normalize_image
    )

    return class_config(
        build_train_dataloader,
        dataset=class_config(
            MultiSampleDataPipe,
            datasets=train_dataset_cfg,
            preprocess_fn=preprocess_transforms,
        ),
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_fn=train_batchprocess_cfg,
        collate_fn=get_callable_cfg(default_collate),
        pin_memory=True,
        shuffle=True,
    )


def get_test_dataloader(
    data_backend: None | ConfigDict,
    image_size: tuple[int, int],
    normalize_image: bool,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for BDD100K tracking."""
    test_dataset = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track_1fps.hdf5",
        keys_to_load=(K.images, K.original_images),
        annotation_path="data/bdd100k/labels/box_track_1fps/val.json",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_1fps_val.pkl",
    )

    # test_dataset = class_config(
    #     BDD100K,
    #     data_root="data/bdd100k/images/track_1fps.hdf5",
    #     keys_to_load=(K.images, K.original_images),
    #     annotation_path="data/bdd100k/labels/box_track_1fps/test.json",
    #     category_map=bdd100k_track_map,
    #     config_path="box_track",
    #     image_channel_mode="BGR",
    #     data_backend=data_backend,
    #     cache_as_binary=True,
    #     cached_file_path="data/bdd100k/pkl/track_1fps_test.pkl",
    # )

    test_preprocess_cfg, test_batchprocess_cfg = get_test_transforms(
        image_size=image_size, normalize_image=normalize_image
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )


def get_bdd100k_track_cfg(
    data_backend: None | ConfigDict = None,
    image_size: tuple[int, int] = (800, 1440),
    normalize_image: bool = False,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for BDD100K tracking."""
    data = DataConfig()

    data.train_dataloader = get_train_dataloader(
        data_backend=data_backend,
        image_size=image_size,
        normalize_image=normalize_image,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        data_backend=data_backend,
        image_size=image_size,
        normalize_image=normalize_image,
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    return data
