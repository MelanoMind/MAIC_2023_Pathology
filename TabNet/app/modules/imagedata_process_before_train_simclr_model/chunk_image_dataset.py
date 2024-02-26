# 작성자 : 박성현
"""
모듈 기능 : data_train_001, .... data_test_084 등으로 분리되어 있는 
여러 pickle 파일 '경로'들을 합친 뒤, chunk_size만큼씩 분리하여 
CustomImageDataLoader 이라는 클래스로 객체화하여 저장한다.
즉 train_001~train_020 을 하나의 CustomImageDataLoader으로, 
train_021~train040 을 하나의 CustomImageDataLoader으로 분리하여 저장한다는 의미.
분리하는 이유는, 이렇게 하지 않으면 SimCLR 훈련 모델에 한 번에 이미지 데이터를 넣어 훈련시키기에
메모리가 부족하기 때문
"""

import pickle
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from app.lib.hydra_setup import setup_config

model_config, path_config, dataset_config, _ = setup_config()


class MILDataset(Dataset):
    def __len__(self):
        return len(self.patch_image_array_list)

    def __getitem__(self, idx):
        slide_idx = self.slide_idx[idx]
        img = self.patch_image_array_list[idx]
        label = self.label_list[idx]

        # # normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        # # img, H, E = normalizer.normalize(I=img, stains=True)

        transform = A.Compose(
            [
                A.Resize(128, 128),
                A.Normalize(
                    mean=[0.8770, 0.7850, 0.8510], std=[0.0924, 0.1323, 0.0980]
                ),  # pkl값을 norm한 결과.
                ToTensorV2(),
            ]
        )
        img = transform(image=np.array(img))["image"]

        return img, label


# CustomImageDataLoader 정의
class CustomImageDataLoader(Dataset):
    """
    loader[0]를 하면 transform된 이미지와 라벨이 나온다.
    """

    def __init__(self):
        self.patch_image_array_list = []
        self.label_list = []

    def __len__(self):
        return len(self.patch_image_array_list)

    def __getitem__(self, idx):
        img = self.patch_image_array_list[idx]
        label = self.label_list[idx]
        transform = A.Compose(
            [
                A.CLAHE(4),
                A.Resize(128, 128),
                A.Normalize(
                    mean=[0.8770, 0.7850, 0.8510], std=[0.0924, 0.1323, 0.0980]
                ),  # pkl값을 norm한 결과.
                ToTensorV2(),
            ]
        )
        img = transform(image=np.array(img))["image"]
        return img, label

    def add_new(self, img, label):
        self.patch_image_array_list.extend(img)
        self.label_list.extend(label)


def save_loader(loader, file_path):
    # file_path에서 디렉토리 경로 추출
    directory = os.path.dirname(file_path)

    # 해당 디렉토리가 존재하지 않는 경우 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 저장
    with open(file_path, "wb") as file:
        pickle.dump(loader, file)


def chunk_wsi_image_dataset():
    # 모든 .pkl 파일의 경로를 가져옵니다.
    train_image_patch_path = path_config["train_image_object"]
    path_for_train_img_pkl = [
        os.path.join(train_image_patch_path, file_name)
        for file_name in os.listdir(train_image_patch_path)
        if file_name.endswith(".pkl")
    ]

    test_image_patch_path = path_config["test_image_object"]
    path_for_test_img_pkl = [
        os.path.join(test_image_patch_path, file_name)
        for file_name in os.listdir(test_image_patch_path)
        if file_name.endswith(".pkl")
    ]

    # 합쳐진 전체 데이터셋 경로 리스트
    all_data_paths = path_for_train_img_pkl + path_for_test_img_pkl

    # 데이터를 n개의 파일로 나누기 위한 청크 크기 계산
    chunk_size = len(all_data_paths) // dataset_config["size_to_split"]
    if len(all_data_paths) % dataset_config["size_to_split"] != 0:
        chunk_size += 1

    if not os.path.exists(path_config["chunked_wsi_images_loader"]):
        os.makedirs(path_config["chunked_wsi_images_loader"], exist_ok=True)

    for i in range(0, len(all_data_paths), chunk_size):
        # 현재 청크에 해당하는 데이터 경로
        current_chunk_paths = all_data_paths[i : i + chunk_size]

        # 현재 청크에 해당하는 CustomImageDataLoader 인스턴스 생성
        custom_loader = CustomImageDataLoader()

        for pkl_file in current_chunk_paths:
            with open(pkl_file, "rb") as file:
                data = pickle.load(file)
                custom_loader.add_new(data.patch_image_array_list, data.label_list)

        # 현재 청크의 CustomImageDataLoader 인스턴스 저장
        save_loader(
            custom_loader,
            os.path.join(
                path_config["chunked_wsi_images_loader"],
                f"chunked_wsi_images_loader_{i // chunk_size}.pkl",
            ),
        )


if __name__ == "__main__":
    chunk_wsi_image_dataset()
