import os
import torch
import pickle
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from app.lib.SimCLR.resnet_simclr.resnet_simclr import ResNetSimCLR
from app.lib.hydra_setup import setup_config

model_config, path_config, dataset_config, simclr_config = setup_config()


class MILDataset(
    Dataset
):  # 참고 https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py
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


def load_model(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load the saved model from a checkpoint.

    :param checkpoint_path: Path to the saved checkpoint.
    :param model: The model (architecture) to load state into.
    :param optimizer: (Optional) The optimizer to load state into.
    :param scheduler: (Optional) The scheduler to load state into.
    :return: The loaded model, optimizer and scheduler.
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, scheduler


# TODO 함수명 바꿔야 할 것 같음.
def iter_all_wsi_images_transform(model, folder_path):
    # 결과를 저장할 사전(dictionary) 초기화
    mean_outputs = {}

    # 폴더 내 모든 .pkl 파일에 대해 반복
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            # 파일 이름에서 'data_'을 제거하여 키 생성
            key = filename.replace("data_", "").replace(".pkl", "")
            # if 'test_001' in key:
            #     test_sample = caluclate_one_wsi_embedding_from_patches(model, file_path)
            #     print(test_sample)
            mean_outputs[key] = caluclate_one_wsi_embedding_from_patches(
                model, file_path
            )
    return mean_outputs


def caluclate_one_wsi_embedding_from_patches(model, file_path):
    # 파일에서 데이터셋 로드
    with open(file_path, "rb") as f:
        test_dataset = pickle.load(f)

    # 데이터셋 길이 확인
    dataset_length = len(test_dataset)

    # 모든 데이터를 하나의 텐서로 결합
    combined_tensor = torch.cat(
        [test_dataset[i][0].unsqueeze(0) for i in range(dataset_length)], dim=0
    )

    # 모델 출력 계산
    model_output = model(combined_tensor)

    # 평균 계산 및 리스트로 변환
    mean_output = torch.mean(model_output, dim=0, keepdim=True)[0].tolist()

    return mean_output


def save_wsi_embbedings(save_path, mean_outputs):
    # 저장 경로가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 결과를 .pkl 파일로 저장
    save_file_path = os.path.join(save_path, "wsi_image_embedding_vectors.pkl")
    with open(save_file_path, "wb") as f:
        pickle.dump(mean_outputs, f)

    print(f"Mean model outputs saved to {save_file_path}")


def load_saved_dict(file_path):
    # 파일에서 사전 로드
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def calculate_patient_embeddings(
    slide_list_data, train_img_embedding_data, test_img_embedding_data
):
    patient_img_embedding_dict = {}

    for patient_id, slide_ids in slide_list_data.items():
        # 각 patient_id에 대한 슬라이드의 임베딩 리스트 초기화
        embeddings = []

        for slide_id in slide_ids:
            # test 또는 train 임베딩 데이터에서 해당 slide_id의 임베딩을 찾음
            if "test" in slide_id:
                embedding = test_img_embedding_data[slide_id]
            else:
                embedding = train_img_embedding_data[slide_id]

            embeddings.append(embedding)

        # 임베딩의 평균 계산
        average_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
        patient_img_embedding_dict[patient_id] = average_embedding

    return patient_img_embedding_dict


def main():
    model = ResNetSimCLR(base_model="resnet18", out_dim=32)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, _ = load_model(
        os.path.join(path_config["best_simclr_model"], "model_best.pth.tar"),
        model,
        optimizer,
    )
    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")

    mean_train_outputs = iter_all_wsi_images_transform(
        model, path_config["train_image_object"]
    )
    save_wsi_embbedings(path_config["train_wsi_image_embedding"], mean_train_outputs)

    mean_test_outputs = iter_all_wsi_images_transform(
        model, path_config["test_image_object"]
    )
    save_wsi_embbedings(path_config["test_wsi_image_embedding"], mean_test_outputs)


if __name__ == "__main__":
    main()
