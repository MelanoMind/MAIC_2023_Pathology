from torchvision.transforms import transforms
from torchvision import transforms, datasets
from app.lib.SimCLR.data_aug.gaussian_blur import GaussianBlur
from app.lib.SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from app.lib.SimCLR.exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
import pickle
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, pickle_file, transform=None):
        # pickle 파일 로드
        with open(pickle_file, "rb") as f:
            self.data_loader = pickle.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, idx):
        img, label = self.data_loader[idx]

        # img가 numpy 배열인 경우
        if isinstance(img, np.ndarray):
            # float32 데이터를 0-255 범위로 스케일링하고 uint8로 변환
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)

            # PIL Image로 변환
            img = Image.fromarray(img)

        # self.transform을 적용 (ContrastiveLearningViewGenerator)
        if self.transform:
            img = self.transform(img)

        return img, label


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # 이 부분을 우리의 프로젝트에 맞추어서 변경해볼 수 있을 것
        data_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def get_dataset(self, name, n_views, file_path):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32), n_views
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                download=True,
            ),
            "custom": lambda: CustomDataset(
                file_path,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(128),  # 예를 들어 96을 사용
                    n_views,
                ),
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
