import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from datetime import datetime
from typing import List, Dict

import pickle
from app.lib.tabnet import *
from app.lib.hydra_setup import setup_config

model_config, path_config, dataset_config, simclr_config = setup_config()


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


def load_saved_dict(file_path):
    # 파일에서 사전 로드
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def calculate_patient_embeddings(
    slide_list_data: Dict, embedding_data: List[int], target: str
):
    patient_img_embedding_dict = {}

    for patient_id, slide_ids in slide_list_data.items():
        embeddings = []

        for slide_id in slide_ids:
            if target in slide_id:
                embedding = embedding_data[slide_id]
                embeddings.append(embedding)

        if embeddings:
            average_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
            patient_img_embedding_dict[patient_id] = average_embedding

    # target을 포함하는 patient_id만 필터링하고 오름차순으로 정렬
    filtered_patient_embeddings = {
        k: v for k, v in sorted(patient_img_embedding_dict.items()) if target in k
    }
    return filtered_patient_embeddings


def train_tabnetmodel():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 이미지 데이터 가져오기
    train_patient_embeddings_dict = load_saved_dict(
        os.path.join(
            path_config["train_patient_average_embeddings"],
            "patient_average_embeddings.pkl",
        )
    )
    test_patient_embeddings_dict = load_saved_dict(
        os.path.join(
            path_config["test_patient_average_embeddings"],
            "patient_average_embeddings.pkl",
        )
    )

    embeddings_train_array = np.array(
        [
            train_patient_embeddings_dict[key]
            for key in train_patient_embeddings_dict.keys()
        ]
    )
    embeddings_test_array = np.array(
        [
            test_patient_embeddings_dict[key]
            for key in test_patient_embeddings_dict.keys()
        ]
    )

    # 테이블 데이터 (metadata) 가져오기
    df = pd.read_csv(path_config["cleaned_metadata"])
    category_columns = [
        "Location",
        "Diagnosis",
        "Growth phase",
        "Level of invasion",
        "Histologic subtype",
        "Tumor cell type",
        "Surgical margin",
        "Lymph node",
        "Precursor lesion",
        "tumor_length_category",
        "tumor_width_category",
        "tumor_height_category",
        "Area of tumor_category",
        "Volume of tumor_category",
        "Breslow thickness_category",
        "Mitosis_category",
    ]
    numerical_columns = [
        "Depth of invasion",
        "Breslow thickness",
        "tumor_length",
        "tumor_width",
        "tumor_height",
        "Area of tumor",
        "Volume of tumor",
        "Mitosis_Value",
    ]

    train, test, category_cols, category_dims, category_indices = TransformDataToTabNet(
        df,
        category_columns,
        numerical_columns,
        model_config["fill_numerical_values_method"],
    ).get_data()

    # 데이터와 레이블 분리
    X, y = train[:, :-1], train[:, -1]

    # X 배열과 embeddings_array를 좌우로 결합
    X_extended = np.concatenate((X, embeddings_train_array), axis=1)
    X = X_extended

    X_test = test[:, :-1]
    X_test_extend = np.concatenate((X_test, embeddings_test_array), axis=1)
    X_test = X_test_extend

    # 먼저, 전체 데이터를 훈련 세트와 임시 테스트 세트로 분할
    X_train_temp, X_sample_test, y_train_temp, y_sample_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 다음으로, 훈련 세트를 다시 훈련 세트와 검증 세트로 분할
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_temp, y_train_temp, test_size=0.25, random_state=42
    )

    # TabNetPretrainer
    unsupervised_model = TabNetPretrainer(
        cat_idxs=category_indices,
        cat_dims=category_dims,
        cat_emb_dim=model_config["cat_emb_dim"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=model_config["lr_unsupervised"]),
        mask_type="entmax",  # "sparsemax",
        n_shared_decoder=1,  # nb shared glu for decoding
        n_indep_decoder=1,  # nb independent glu for decoding
        grouped_features=[
            [9, 10, 11, 12, 13, 18, 19, 20, 21, 22],
            [15, 23],
            [14, 17],
            list(range(24, 39)),
        ],
        verbose=model_config["verbose"],
    )

    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        max_epochs=model_config["max_epochs"],
        patience=model_config["patience"],
        batch_size=model_config["batch_size"],
        virtual_batch_size=model_config["virtual_batch_size"],
        num_workers=0,
        drop_last=False,
        pretraining_ratio=model_config["pretraining_ratio"],
    )

    # Make reconstruction from a dataset
    reconstructed_X, embedded_X = unsupervised_model.predict(X_valid)
    assert reconstructed_X.shape == embedded_X.shape

    unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(
        X_valid
    )

    path_to_save = os.path.join(
        path_config["tabnet_pretrained_model"], f"{current_time}/model_{current_time}"
    )
    print(path_to_save)
    path_to_load = os.path.join(
        path_config["tabnet_pretrained_model"],
        f"{current_time}/model_{current_time}.zip",
    )
    unsupervised_model.save_model(path_to_save)
    loaded_pretrain = TabNetPretrainer()
    loaded_pretrain.load_model(path_to_load)

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=model_config["lr_trained"]),
        scheduler_params={
            "step_size": 10,  # how to use learning rate scheduler
            "gamma": 0.9,
        },
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",  # This will be overwritten if using pretrain model
        cat_emb_dim=model_config["cat_emb_dim"],
        cat_idxs=category_indices,
        cat_dims=category_dims,
        grouped_features=[[9, 10, 11, 12, 13, 18, 19, 20, 21, 22], [15, 23], [14, 17]],
        verbose=model_config["verbose"],
        device_name="cuda",
    )
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["auc"],
        max_epochs=model_config["max_epochs"],
        patience=model_config["patience"],
        batch_size=model_config["batch_size_training"],
        virtual_batch_size=model_config["virtual_batch_size_training"],
        num_workers=0,
        weights=1,
        drop_last=False,
        from_unsupervised=loaded_pretrain,
    )

    preds_to_sample_test = clf.predict_proba(X_valid)[:, 1]
    print(f"fold : ", roc_auc_score(y_valid, preds_to_sample_test))

    preds = clf.predict_proba(X_test)

    os.makedirs(
        os.path.join(path_config["tabnet_trained_model"], f"{current_time}"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(path_config["tabnet_prediction_result"], f"{current_time}"),
        exist_ok=True,
    )
    clf.save_model(
        os.path.join(
            path_config["tabnet_trained_model"], f"{current_time}/model_{current_time}"
        )
    )
    np.save(
        os.path.join(
            path_config["tabnet_prediction_result"],
            f"{current_time}",
            f"model_{current_time}_preds.npy",
        ),
        preds[:, 1],
    )


if __name__ == "__main__":
    train_tabnetmodel()
