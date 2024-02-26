import os

# 모델 설정 데이터 클래스 정의
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import albumentations as A
from argparse import Namespace
from app.lib.SimCLR.data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset,
)
from app.lib.SimCLR.resnet_simclr.resnet_simclr import ResNetSimCLR
from app.lib.SimCLR.simclr import SimCLR
from app.lib.hydra_setup import setup_config

model_config, path_config, dataset_config, simclr_config = setup_config()


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

        # 이미지가 uint8 형식인지 확인하고, 아니라면 변환
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        transform = A.Compose(
            [
                A.CLAHE(clip_limit=2),
                A.Resize(128, 128),
                A.Normalize(
                    mean=[0.8770, 0.7850, 0.8510], std=[0.0924, 0.1323, 0.0980]
                ),
                # ToTensorV2(),
            ]
        )

        img = transform(image=img)["image"]
        return img, label

    def add_new(self, img, label):
        self.patch_image_array_list.extend(img)
        self.label_list.extend(label)


def main():
    args = Namespace(
        data="./datasets",
        dataset_name="custom",
        arch="resnet18",
        workers=12,
        epochs=1,
        batch_size=64,
        lr=0.0009,
        weight_decay=1e-4,
        seed=None,
        disable_cuda=False,
        fp16_precision=False,
        out_dim=simclr_config["out_dim"],
        log_every_n_steps=1,
        temperature=0.07,
        n_views=2,
        gpu_index=0,
        log_dir=path_config["train_log"],
    )
    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(path_config["best_simclr_model"]):
        os.makedirs(path_config["best_simclr_model"], exist_ok=True)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)
    # size_to_split개의 custom_loader 파일에 대한 경로 리스트 생성
    custom_loader_paths = [
        os.path.join(
            path_config["chunked_wsi_images_loader"],
            f"chunked_wsi_images_loader_{i}.pkl",
        )
        for i in range(dataset_config["size_to_split"])
    ]
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    # 전체 에포크 수를 T_max로 사용
    total_epochs = len(custom_loader_paths)

    # 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=0, last_epoch=-1
    )

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

    for loader_path in custom_loader_paths:
        # 훈련 데이터셋 로드
        train_dataset = dataset.get_dataset(
            args.dataset_name, args.n_views, loader_path
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        # 모델 훈련
        with torch.cuda.device(args.gpu_index):
            simclr.train(train_loader)


if __name__ == "__main__":
    main()
