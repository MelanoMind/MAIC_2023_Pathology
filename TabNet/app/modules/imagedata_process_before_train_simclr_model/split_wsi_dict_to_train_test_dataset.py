# 작성자 : 박성현
"""
모듈 기능 : downloadAndSaveWSIImageIntoOneDictionary.py를 통해 생성된 patch_dictionary_by_wsi
데이터를, train, test로 나누어 train_imageset, test_imageset 폴더에 
각각의 WSI (patch_image_array_list)를 저장한다.
"""


import json
from app.lib.hydra_setup import setup_config
from dataclasses import dataclass
from typing import List, Any
import pickle
import os
from torch.utils.data import Dataset
from multiprocessing import Pool
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
model_config, path_config, dataset_config, _ = setup_config()


class MILDataset(Dataset):
    def __init__(self, patch_image_array_list, slide_list, slide_idx, label_list):
        self.patch_image_array_list = patch_image_array_list
        self.slide_list = slide_list
        self.slide_idx = slide_idx
        self.label_list = label_list

    def __len__(self):
        return len(self.patch_image_array_list)

    def __getitem__(self, idx):
        img = self.patch_image_array_list[idx]
        label = self.label_list[idx]
        return img, label


def save_dataset(args):
    key, value, path_config = (
        args  # 멀티프로세싱 pool.map 사용 시, args 튜플을 풀어서 사용
    )

    logging.info(f"Processing {key}...")

    dataset = MILDataset(**value)

    # 저장 경로 설정
    if "train" in key:
        save_dir = path_config["train_image_object"]
    elif "test" in key:
        save_dir = path_config["test_image_object"]

    # 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"data_{key}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    logging.info(f"Saved {key} to {save_path}")


def split_wsi_dict_to_train_test_dataset():
    # path_config에서 JSON 파일의 경로를 가져옵니다.
    json_slide_list_file_path = path_config["slide_list_by_patient"]
    json_recurrence_file_path = path_config["recurrence_by_patient"]
    with open(json_slide_list_file_path, "r") as f:
        slide_list_by_patient = json.load(f)

    with open(json_recurrence_file_path, "r") as f:
        recurrence_dict = json.load(f)

    for key, values in recurrence_dict.items():
        if "train" in key:
            # NaN 값을 처리하기 위해 np.nan을 None으로 변환하고, float를 int로 변환
            recurrence_dict[key] = [
                int(value) if value == value else None for value in values
            ]

    inverse_dict = {}
    for pid, test_list in slide_list_by_patient.items():
        for test_id in test_list:
            inverse_dict[test_id] = pid

    with open(
        os.path.join(path_config["image_dataset"], "patch_dictionary_by_wsi.pkl"), "rb"
    ) as f:
        patch_dict = pickle.load(f)

    # 멀티프로세싱을 사용하여 각 데이터셋을 병렬로 저장
    args_list = [(key, value, path_config) for key, value in patch_dict.items()]
    with Pool(processes=16) as pool:  # 사용할 프로세스 수 지정
        pool.map(save_dataset, args_list)


if __name__ == "__main__":
    split_wsi_dict_to_train_test_dataset()
