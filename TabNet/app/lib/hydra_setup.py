from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose
from omegaconf import OmegaConf
from typing import Any


# 모델 설정 데이터 클래스 정의
@dataclass
class ModelConfig:
    n_splits: int
    max_epochs: int
    patience: int
    fill_numerical_values_method: str
    folding_strategy: str
    random_state: int
    cat_emb_dim: int
    optimizer_fn: str
    lr_unsupervised: float
    lr_trained: float
    n_shared_decoder: int
    n_indep_decoder: int
    batch_size: int
    batch_size_training: int
    virtual_batch_size: int
    virtual_batch_size_training: int
    pretraining_ratio: float
    weights: Any
    verbose: int


# 경로 설정 데이터 클래스 정의
@dataclass
class PathConfig:
    metadata_dir: str
    processed_dataset: str
    cleaned_metadata: str
    slide_list_by_patient: str
    recurrence_by_patient: str
    image_dataset: str
    train_image_object: str
    test_image_object: str
    chunked_wsi_images_loader: str
    train_log: str
    best_simclr_model: str
    train_wsi_image_embedding: str
    test_wsi_image_embedding: str
    train_patient_average_embeddings: str
    test_patient_average_embeddings: str
    tabnet_pretrained_model: str
    tabnet_trained_model: str
    tabnet_prediction_result: str
    model_dir: str
    dataset_dir: str


# 데이터셋 설정 데이터 클래스 정의
@dataclass
class SimCLRConfig:
    out_dim: int


# 데이터셋 설정 데이터 클래스 정의
@dataclass
class DataSetConfig:
    size_to_split: int


def setup_config():
    # Hydra 구성 저장소에 모델 설정 추가
    cs = ConfigStore.instance()
    cs.store(group="model", name="config", node=ModelConfig)
    cs.store(group="path", name="config", node=PathConfig)
    cs.store(group="dataset", name="config", node=DataSetConfig)
    cs.store(group="simclr_model", name="config", node=SimCLRConfig)

    # Hydra 초기화 및 설정 구성
    with initialize(config_path=".", version_base="1.1"):
        cfg = compose(config_name="config")

    # OmegaConf를 사용하여 설정을 딕셔너리로 변환
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # 개별 설정 딕셔너리를 반환
    return (
        config_dict["model"],
        config_dict["path"],
        config_dict["dataset"],
        config_dict["simclr_model"],
    )
