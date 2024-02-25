import pandas as pd
from matplotlib import pyplot as plt
from dataprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# 모델 설정 데이터 클래스 정의
@dataclass
class ModelConfig:
    n_splits: int
    max_epochs: int
    patience: int
    fill_numerical_values_method: str
    folding_strategy : str
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
    model_dir: str
    dataset_dir: str
    original_dataFrame_dir: str
    path_to_save_pretrained_model : str
    path_to_save_trained_model : str
    path_to_save_prediction_result : str

# Hydra 구성 저장소에 모델 설정 추가
cs = ConfigStore.instance()
cs.store(group="model", name="config", node=ModelConfig)
cs.store(group="path", name="config", node=PathConfig)

def save_model(model, fold, current_time, model_type="semi_pretrain_model"):
    # 저장할 경로 설정
    base_path = os.path.join("tabnetmodels", model_type, current_time)
    
    # 디렉터리가 존재하지 않으면 생성
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # 모델 저장 경로 설정
    if model_type == "semi_pretrain_model":
        save_path = os.path.join(base_path, f"test_pretrain{fold+1}.zip")
        model.save_model(save_path)
    else:
        save_path = os.path.join(base_path, f"model{fold+1}.pt")
        torch.save(model.state_dict(), save_path)
    
    print(f"Model saved to {save_path}")
    return save_path  # 저장 경로 반환

# Hydra 데코레이터 사용
@hydra.main(config_path="/home/admin/seonghyun", config_name="config")
def main(cfg:DictConfig):
    model_config = cfg.model
    path_config = cfg.path
    df = pd.read_csv(path_config.original_dataFrame_dir)
    category_columns = [
                'Location', 'Diagnosis', 'Growth phase', 'Level of invasion', 
                'Histologic subtype', 'Tumor cell type', 'Surgical margin', 'Lymph node', 
                'Precursor lesion', 'tumor_length_category', 'tumor_width_category',
                'tumor_height_category', 'Area of tumor_category',
                'Volume of tumor_category', 'Breslow thickness_category',
                'Mitosis_category'
            ]
    numerical_columns = ['Depth of invasion', 'Breslow thickness',
        'tumor_length', 'tumor_width', 'tumor_height', 'Area of tumor',
        'Volume of tumor', 'Mitosis_Value']

    # 스크립트 시작 시간 기록
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    train, test, category_cols, category_dims, category_indices = TransformDataToTabNet(df, category_columns, numerical_columns, model_config.fill_numerical_values_method).get_data()

    # 데이터와 레이블 분리 
    X, y  = train[:, :-1], train[:, -1]
    X_test = test[:, :]

    # KFold 또는 StratifiedKFold 객체 생성
    if model_config.folding_strategy == "kf":
        kf = KFold(n_splits=model_config.n_splits, shuffle=True, random_state=model_config.random_state)
    elif model_config.folding_strategy == "stratified":
        kf = StratifiedKFold(n_splits=model_config.n_splits, shuffle=True, random_state=model_config.random_state)
    else:
        raise ValueError(f"Unknown folding strategy: {model_config.folding_strategy}")

    # 각 fold에 대해 반복
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y) if model_config.folding_strategy == "stratified" else kf.split(X)):
        print(f"Training fold {fold+1}/{model_config.n_splits}")
        
        # 훈련 데이터와 검증 데이터 분할
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_valid_fold, y_valid_fold = X[valid_idx], y[valid_idx]

        # TabNetPretrainer
        unsupervised_model = TabNetPretrainer(
            cat_idxs=category_indices,
            cat_dims=category_dims,
            cat_emb_dim=model_config.cat_emb_dim,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=model_config.lr_unsupervised),
            mask_type='entmax', # "sparsemax",
            n_shared_decoder=1, # nb shared glu for decoding
            n_indep_decoder=1, # nb independent glu for decoding
            grouped_features=[[9,10,11,12,13,18,19,20,21,22], [15,23], [14,17]],
            verbose=model_config.verbose,
        )

        unsupervised_model.fit(
            X_train=X_train_fold,
            eval_set=[X_valid_fold],
            max_epochs=model_config.max_epochs , patience=model_config.patience,
            batch_size=model_config.batch_size, virtual_batch_size=model_config.virtual_batch_size,
            num_workers=0,
            drop_last=False,
            pretraining_ratio=model_config.pretraining_ratio,
        )

        # Make reconstruction from a dataset
        reconstructed_X, embedded_X = unsupervised_model.predict(X_valid_fold)
        assert(reconstructed_X.shape==embedded_X.shape)

        unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(X_valid_fold)

        path_to_save = os.path.join(path_config.path_to_save_pretrained_model, f"{current_time}/model_{fold+1}")
        path_to_load = os.path.join(path_config.path_to_save_pretrained_model, f"{current_time}/model_{fold+1}.zip")
        unsupervised_model.save_model(path_to_save)
        loaded_pretrain = TabNetPretrainer()
        loaded_pretrain.load_model(path_to_load)

        clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params = dict(lr=model_config.lr_trained),
                       scheduler_params = {"step_size":10, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn = torch.optim.lr_scheduler.StepLR,
                       mask_type = 'entmax', # This will be overwritten if using pretrain model
                       cat_emb_dim = model_config.cat_emb_dim,
                       cat_idxs = category_indices,
                       cat_dims = category_dims,
                       grouped_features= [[9,10,11,12,13,18,19,20,21,22], [15,23], [14,17]],
                       verbose=model_config.verbose, 
                      )
        clf.fit(
            X_train=X_train_fold, y_train=y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_valid_fold, y_valid_fold)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=model_config.max_epochs , patience=model_config.patience,
            batch_size=model_config.batch_size_training, virtual_batch_size=model_config.virtual_batch_size_training,
            num_workers=0,
            weights=1,
            drop_last=False,
            from_unsupervised=loaded_pretrain,
        )
        
        preds = clf.predict_proba(X_test)
        # 저장할 기본 디렉토리를 생성합니다.
        os.makedirs(os.path.join(path_config.path_to_save_trained_model, f'{current_time}'), exist_ok=True)
        os.makedirs(os.path.join(path_config.path_to_save_prediction_result,f"{current_time}"), exist_ok=True)
        np.save(os.path.join(path_config.path_to_save_prediction_result,f"{current_time}",f"model_{fold+1}_preds.npy"), preds[:,1])
        clf.save_model(os.path.join(path_config.path_to_save_trained_model, f"{current_time}/model_{fold+1}"))
        
if __name__=='__main__':
    main()