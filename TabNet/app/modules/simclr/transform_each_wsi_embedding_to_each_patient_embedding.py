import pickle
import json
from typing import Dict
import os
import numpy as np
from app.lib.hydra_setup import setup_config
from typing import List

model_config, path_config, dataset_config, _ = setup_config()


def load_pickle_file(file_path: str) -> Dict:
    # 파일에서 사전 로드
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def load_slide_list_by_patient(file_path: str) -> json:
    with open(file_path, "r") as file:
        slide_list_by_patient = json.load(file)
    return slide_list_by_patient


def save_average_embeddings(path, embeddings):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "patient_average_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)


def calculate_patient_average_embeddings(
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
    ordered_patient_embeddings = {
        k: v for k, v in sorted(patient_img_embedding_dict.items()) if target in k
    }
    return ordered_patient_embeddings


# 이 모듈은 lib으로 올라가야할 것 같은데.
def get_and_save_patient_average_embedding_dict():
    train_img_data_path = os.path.join(
        path_config["train_wsi_image_embedding"], "wsi_image_embedding_vectors.pkl"
    )
    train_img_embedding_data = load_pickle_file(train_img_data_path)

    test_img_data_path = os.path.join(
        path_config["test_wsi_image_embedding"], "wsi_image_embedding_vectors.pkl"
    )
    test_img_embedding_data = load_pickle_file(test_img_data_path)

    slide_list_path = path_config["slide_list_by_patient"]
    slide_list_data = load_slide_list_by_patient(slide_list_path)

    ordered_train_patient_embeddings = calculate_patient_average_embeddings(
        slide_list_data, train_img_embedding_data, "train"
    )
    ordered_test_patient_embeddings = calculate_patient_average_embeddings(
        slide_list_data, test_img_embedding_data, "test"
    )

    save_average_embeddings(
        path_config["train_patient_average_embeddings"],
        ordered_train_patient_embeddings,
    )
    save_average_embeddings(
        path_config["test_patient_average_embeddings"], ordered_test_patient_embeddings
    )

    return ordered_train_patient_embeddings, ordered_test_patient_embeddings


if __name__ == "__main__":
    train_patient_embeddings, test_patient_embeddings = (
        get_and_save_patient_average_embedding_dict()
    )
