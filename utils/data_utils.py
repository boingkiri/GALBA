import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision

import random
from .dataset.classification import ClassificationDataset

from lightning.pytorch import LightningDataModule

import os
import math
import json
import torchvision.transforms as transforms
from typing import Tuple
from PIL import Image 

import sys
import subprocess
import time

import datasets
from datasets import Image as HFImage
import yaml
from tqdm import tqdm
# from datasets import Image as HFDataset



try:
    import torch_xla
    import torch_xla.runtime as xr
    import webdataset as wds
    import io
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
    GCS_BUCKET_NAME=os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
except Exception:
    _HAS_XLA = False


def get_class_idx_to_class_name_mapping(cfg):
    if cfg.dataset.name == "imagenet_100":
        # Same class names (and order) as the `label` feature of the HF dataset "clane9/imagenet-100".
        with open(resolve_project_path(cfg.dataset.imagenet_100_idx_label_mapping_path), 'r') as f:
            label_names = yaml.safe_load(f)
        return {i: label_names[i].split(",") for i in range(len(label_names))}
    elif cfg.dataset.name == "imagenette":
        tmp_dataset = torchvision.datasets.Imagenette(root=resolve_project_path(cfg.dataset.dataset_path), split='train', download=True)
        return {i: class_name for i, class_name in enumerate(tmp_dataset.classes)}
    
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} is not supported for class idx to name mapping.")

def get_gcs_shards(gcs_pattern):
    """
    Get list of GCS shards using gsutil.
    """
    try:
        # Use gsutil to list files matching the pattern
        result = subprocess.run(["gsutil", "ls", gcs_pattern], capture_output=True, text=True, check=True)
        shards = result.stdout.strip().split('\n')
        return [s for s in shards if s.strip()] # Filter empty strings
    except subprocess.CalledProcessError as e:
        print(f"Error listing GCS shards: {e}", file=sys.stderr)
        return []

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_project_path(path):
    """
    Resolve `path` to an absolute path.
    - Absolute paths are returned unchanged.
    - Relative paths are resolved against the project root (the directory
      containing `utils/`), so paths stay valid even when Hydra changes the CWD.
    """
    if path is None:
        return path
    path = os.fspath(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))









def identity(x):
    return x

def rename_keys(sample):
    """
    Rename keys to standard names if necessary.
    Handles 'png.jpg' -> 'jpg' and 'png.cls' -> 'cls'.
    """
    if 'png.jpg' in sample:
        sample['jpg'] = sample.pop('png.jpg')
    if 'png.cls' in sample:
        sample['cls'] = sample.pop('png.cls')
    return sample

def convert_rgb(img):
    return img.convert("RGB")

def get_rank_and_world_size():
    """
    PL(TPUStrategy), Accelerate, torchrun 환경에서 
    공통적으로 Rank와 World Size를 가져옵니다.
    """
    # 1. 환경 변수 확인 (가장 안전 - 장치 직접 접근 없음)
    # PL과 Accelerate(torchrun) 모두 RANK, WORLD_SIZE를 설정합니다.
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")

    # print(f"Rank: {rank}, World Size: {world_size}", file=sys.stderr)

    if rank is None or world_size is None:
        rank = xr.global_ordinal()
        world_size = xr.world_size()
    else:
        rank = int(rank)
        world_size = int(world_size)
    
    return rank, world_size

def my_splitter(src):
    urls = list(src)
    rank, world_size = get_rank_and_world_size()

    my_shards = urls[rank::world_size]
    print(f"[Rank {rank}/{world_size}] I took {len(my_shards)} shards: {my_shards}", file=sys.stderr)
    
    yield from my_shards


def ensure_rgb(sample):
    """WebDataset 샘플의 이미지를 RGB로 변환합니다."""
    if "jpg" in sample:
        sample["jpg"] = sample["jpg"].convert("RGB")
    elif "png" in sample:
        # 키 이름이 png일 경우 처리
        sample["png"] = sample["png"].convert("RGB")
    return sample

class ImageDatasetForScoring(Dataset):
    def __init__(self, dataset_dir: str, transform):
        self.image_dir = os.path.join(dataset_dir, "images")
        self.label_path = os.path.join(dataset_dir, "labels.json")

        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        self.transform = transform
        self.image_files = sorted(list(self.labels.keys()))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, int]:
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)  # [C, H, W] torch.Tensor
        label = self.labels[image_name]

        return image_name, img_tensor, label

class HFAsTorchDataset(Dataset):
    """HF Dataset을 (tensor, int) 튜플로 내보내는 어댑터"""
    def __init__(self, hf_ds):
        self.ds = hf_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]             # {"image": Tensor, "label": int}
        return ex["image"], int(ex["label"])

class ApplyTransformBatch:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        # batch: {"image": [PIL or np...], "label": [int, ...]}
        imgs_out = []
        for img in batch["image"]:
            
            # 흑백 대비
            # if hasattr(img, "convert"):
            #     img = img.convert("RGB")

            # torchvision Compose는 단일 이미지에 적용
            img = self.transform(img)
            imgs_out.append(img)
        # 반드시 "시퀀스(dict of sequences)"로 반환!
        return {"image": imgs_out, "label": batch["label"]}






class DataModule(LightningDataModule):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__()
        # Save config and hyperparameters. `save_hyperparameters` makes them accessible
        # via self.hparams and automatically logs them.
        self.cfg = cfg # Keep a direct reference for convenience
        self.num_processes = num_processes
        self.seed = seed
        
        # These will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # A list to hold all datasets that are concatenated for training
        self._train_datasets_to_concat = []
        self.real_dataset_loaded = False
        
        # A dataset for computing guiding metrics if enabled
        self.holdout_subset = None
        gradient_accumulation_steps = self.cfg.train.get("gradient_accumulation_steps", 1)
        self.batch_size = self.cfg.train.batch_size // self.num_processes // gradient_accumulation_steps
        print(f"Batch size per process: {self.batch_size}", file=sys.stderr)

        if cfg.train.get("mixup_alpha", 0) > 0 or cfg.train.get("cutmix_alpha", 0) > 0:
            self.collate_fn = self._collate_fn_aug
            print("get collate_fn for mixup/cutmix augmentation", file=sys.stderr)
        else:
            self.collate_fn = None

        # self.cfg.train.num_workers = 0 # TMP

    # This hook is called once on the main process to download data.
    def prepare_data(self):
        """
        Download datasets if needed. This is called on only one GPU.
        """
        print("DataModule: prepare_data()")
        if self.cfg.dataset.name == 'cifar10':
            torchvision.datasets.CIFAR10(root=self.cfg.dataset.dataset_path, train=True, download=True)
            torchvision.datasets.CIFAR10(root=self.cfg.dataset.dataset_path, train=False, download=True)
        elif self.cfg.dataset.name == 'mnist':
            torchvision.datasets.MNIST(root=self.cfg.dataset.dataset_path, train=True, download=True)
            torchvision.datasets.MNIST(root=self.cfg.dataset.dataset_path, train=False, download=True)
        elif self.cfg.dataset.name  == 'imagenette':
            torchvision.datasets.Imagenette(root=resolve_project_path(self.cfg.dataset.dataset_path), split='train', download=True)
            torchvision.datasets.Imagenette(root=resolve_project_path(self.cfg.dataset.dataset_path), split='val', download=True)
        elif self.cfg.dataset.name == 'imagenet_100':
            datasets.load_dataset("clane9/imagenet-100")

        print("DataModule: prepare_data() done.")

        
        # For ImageNet, we assume it's already downloaded.

    # This hook is called on every GPU.
    def setup(self, stage: str):
        """
        Assign train/val/test datasets for dataloaders.
        - `stage` can be 'fit', 'validate', 'test', or 'predict'.
        """
        print(f"DataModule: setup(stage='{stage}')")
        random.seed(self.seed)
        if stage == 'fit' or stage is None:
            # Setup training data
            # train_transforms = self._create_transform(train_mode=True)
            train_transforms = self._create_transform(transform_mode="train")
            num_sample = self.cfg.num_initial_labeled_real_samples # TODO: ugly code

            full_train_dataset = None
            if num_sample == 0:
                print("No initial labeled samples specified. Skip loading the initial labeled samples")
            else:
                full_train_dataset = self._load_initial_dataset(
                    root=self.cfg.dataset.dataset_path, train_mode=True, transform=train_transforms)
                
                num_sample = self._get_initial_sample_count(full_train_dataset)
                # Handle initial subsetting of the labeled data
                # if not self._train_datasets_to_concat: # Only do this once
                if self.cfg.num_initial_labeled_real_samples > 0 and not self.real_dataset_loaded:
                    self.real_dataset_loaded = True
                    self.selected_initial_indices_for_real_dataset = random.sample(range(len(full_train_dataset)), num_sample)
                    initial_labeled_real_subset = torch.utils.data.Subset(
                        full_train_dataset, self.selected_initial_indices_for_real_dataset)
                    self.initial_labeled_real_subset = initial_labeled_real_subset
                    self.add_dataset(initial_labeled_real_subset)
                    print(f"Initial labeled training dataset created: {len(initial_labeled_real_subset)} samples")

            if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
                full_train_dataset = self._load_initial_dataset(
                    root=self.cfg.dataset.dataset_path, train_mode=True, transform=train_transforms) \
                        if full_train_dataset is None else full_train_dataset
                sampling_range = set(range(len(full_train_dataset))) - set(self.selected_initial_indices_for_real_dataset) \
                    if hasattr(self, "selected_initial_indices_for_real_dataset") else set(range(len(full_train_dataset)))
                self.selected_initial_indices_for_holdout_dataset = random.sample(
                    list(sampling_range),
                    self.cfg.dataset.num_holdout_samples
                )
                self.holdout_subset = torch.utils.data.Subset(
                    full_train_dataset, self.selected_initial_indices_for_holdout_dataset)
                print(f"Holdout dataset created: {len(self.holdout_subset)} samples")
                # print(f"Index: {self.selected_initial_indices_for_holdout_dataset}", file=sys.stderr)

            # Setup validation data
            # val_transforms = self._create_transform(train_mode=False)
            val_transforms = self._create_transform(transform_mode="test")
            self.val_dataset = self._load_initial_dataset(root=self.cfg.dataset.dataset_path, train_mode=False, transform=val_transforms)
            print(f"Validation dataset loaded: {len(self.val_dataset)} samples")

        # if stage == 'test' or stage is None:
        if stage == "test":
            test_transforms = self._create_transform(transform_mode="test")
            self.test_dataset = self._load_initial_dataset(root=self.cfg.dataset.dataset_path, train_mode=False, transform=test_transforms)
            print(f"Test dataset loaded: {len(self.test_dataset)} samples")
        
        if stage == "predict":
            test_transforms = self._create_transform(transform_mode="test")
            # self.predict_dataset = self._load_initial_dataset(root=self.cfg.dataset.dataset_path, train_mode=False, transform=test_transforms)
            self.predict_dataset = self._load_initial_dataset(root=self.cfg.dataset.dataset_path, train_mode=True, transform=test_transforms)
            print(f"Predict dataset loaded: {len(self.predict_dataset)} samples")


    def add_dataset(self, dataset: Dataset):
        """
        Adds a new dataset (e.g., from a generator) to the training set.
        """
        # if not isinstance(dataset, Dataset):
        #     raise TypeError("Input must be a torch.utils.data.Dataset")
        
        self._train_datasets_to_concat.append(dataset)
        
        # Check if any dataset is an IterableDataset (like WebDataset)
        has_iterable = any(isinstance(d, torch.utils.data.IterableDataset) for d in self._train_datasets_to_concat)
        
        if has_iterable:
            # If we mix Map and Iterable, we must treat all as Iterable (ChainDataset)
            # Convert MapDatasets to IterableDatasets if needed (not trivial for random access, but for training it's fine)
            # However, torch's ChainDataset expects all IterableDatasets. 
            # A MapDataset is NOT an IterableDataset by inheritance usually (it implements __getitem__).
            # We can rely on a helper or just try ChainDataset if they are compatible.
            # Actually, standard practice: wrap map dataset into iterable.
            
            class IterableWrapper(torch.utils.data.IterableDataset):
                def __init__(self, map_dataset):
                    self.map_dataset = map_dataset

                def __iter__(self):
                    # 1. 분산 학습 정보 가져오기 (TPU 코어/프로세스 단위)
                    import torch.distributed as dist
                    if dist.is_initialized():
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                    else:
                        world_size = 1
                        rank = 0

                    # 2. DataLoader 워커 정보 가져오기 (프로세스 내부 스레드 단위)
                    worker_info = torch.utils.data.get_worker_info()
                    
                    # 전체 인덱스 생성
                    indices = list(range(len(self.map_dataset)))
                    
                    # [수정 핵심] Rank별로 먼저 인덱스를 나눔
                    per_rank = int(math.ceil(len(indices) / float(world_size)))
                    rank_indices = indices[rank * per_rank : min((rank + 1) * per_rank, len(indices))]
                    
                    if worker_info is None:
                        for i in rank_indices:
                            yield self.map_dataset[i]
                    else:
                        # Rank에 할당된 인덱스를 다시 워커별로 나눔
                        per_worker = int(math.ceil(len(rank_indices) / float(worker_info.num_workers)))
                        worker_id = worker_info.id
                        iter_start = worker_id * per_worker
                        iter_end = min(iter_start + per_worker, len(rank_indices))
                        
                        for i in range(iter_start, iter_end):
                            yield self.map_dataset[rank_indices[i]]

            iterable_datasets = []
            for d in self._train_datasets_to_concat:
                if isinstance(d, torch.utils.data.IterableDataset):
                    iterable_datasets.append(d)
                    print("Adding IterableDataset to ChainDataset", file=sys.stderr)
                else:
                    iterable_datasets.append(IterableWrapper(d))
                    print("Wrapping MapDataset to IterableDataset for ChainDataset", file=sys.stderr)
            
            self.train_dataset = torch.utils.data.ChainDataset(iterable_datasets)
            # Note: len(self.train_dataset) will fail for ChainDataset/WebDataset. 
            # We should probably handle that in logging or wherever len() is called.
            print(f"New dataset added (ChainDataset). Total datasets: {len(iterable_datasets)}")

        else:
            self.train_dataset = ConcatDataset(self._train_datasets_to_concat)
            print(f"New dataset added. Total training samples: {len(self.train_dataset)}")
    
    def add_generated_dataset_from_tar_urls(self, tar_urls: list):
        """
        A specific helper to add a dataset of generated images from a list of tar URLs.
        """
        # Use the same transforms as the training set
        # transform = self._create_transform(train_mode=True)
        transform = self._create_transform(transform_mode="train")
        
        if _HAS_XLA:
            # TPU VM environment: Load from GCS using WebDataset
            print(f"Loading generated dataset from GCS tar URLs", file=sys.stderr)
            
            print("wds.utils.pytorch_worker_info()", wds.utils.pytorch_worker_info(), file=sys.stderr)
            
            dataset = (
                wds.WebDataset(tar_urls, nodesplitter=my_splitter, shardshuffle=True, empty_check=False)
                .decode("pil")
                .map(ensure_rgb)
                .shuffle(5000) # 충분한 버퍼 크기
                .to_tuple("jpg", "cls")
                .map_tuple(transform, identity)
            )
            self.add_dataset(dataset)
        else:
            raise NotImplementedError("Non-TPU environment for tar URL based generated dataset is not implemented.")
        
        print(f"Generated dataset added from tar URLs")

    def add_generated_dataset(self, root: str):
        """
        A specific helper to add a dataset of generated images from a directory.
        """
        image_dir = os.path.join(root, 'images')
        label_path = os.path.join(root, 'labels.json')
        # Use the same transforms as the training set
        # transform = self._create_transform(train_mode=True)
        transform = self._create_transform(transform_mode="train")
        
        if _HAS_XLA:
            # TPU VM environment: Load from GCS using WebDataset
            # gcs_image_dir = f"gs://generative_active_learning/{image_dir}" if "gs://" not in image_dir else image_dir
            gcs_image_dir = f"gs://{GCS_BUCKET_NAME}/{image_dir}" if "gs://" not in image_dir else image_dir
            print(f"Loading generated dataset from GCS: {gcs_image_dir}", file=sys.stderr)
            
            # Find all tar shards
            shards = get_gcs_shards(f"{gcs_image_dir}/*.tar")
            if not shards:
                print(f"No shards found at {gcs_image_dir}", file=sys.stderr)
                return

            print(f"Found {len(shards)} shards.", file=sys.stderr)
            
            # Create WebDataset
            # dataset = (
            #     wds.WebDataset(shards, nodesplitter=wds.split_by_node, shardshuffle=True)
            #     .compose(wds.split_by_worker)
            #     .decode("pil")
            #     .map(ensure_rgb)
            #     .shuffle(5000) # 충분한 버퍼 크기
            #     .to_tuple("jpg", "cls")
            #     .map_tuple(transform, identity)
            # )
            dataset = (
                wds.WebDataset(shards, nodesplitter=my_splitter, shardshuffle=True)
                .decode("pil")
                .map(ensure_rgb)
                .shuffle(5000) # 충분한 버퍼 크기
                .to_tuple("jpg", "cls")
                .map_tuple(transform, identity)
            )
            self.add_dataset(dataset)
        else:
            # Replace with your actual ClassificationDataset
            dataset = ClassificationDataset(
                cfg=self.cfg,
                data_dir=image_dir,
                label_path=label_path,
                transform=transform
            )
            self.add_dataset(dataset)
        print(f"Generated dataset added from {root}")

    
    def _collate_fn_aug(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)

        mixup_alpha = self.cfg.train.get("mixup_alpha", 0)
        cutmix_alpha = self.cfg.train.get("cutmix_alpha", 0)

        if mixup_alpha > 0 and cutmix_alpha > 0:
            use_mixup = random.random() < 0.5
            if use_mixup:
                images, labels = self.mixup(images, labels, mixup_alpha)
            else:
                images, labels = self.cutmix(images, labels, cutmix_alpha)
        elif mixup_alpha > 0:
            images, labels = self.mixup(images, labels, mixup_alpha)
        elif cutmix_alpha > 0:
            images, labels = self.cutmix(images, labels, cutmix_alpha)

        return images, labels

    def mixup(self, images, labels, alpha):
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
        rand_index = torch.randperm(images.size(0))
        shuffled_images = images[rand_index]
        shuffled_labels = labels[rand_index]

        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        labels_one_hot = F.one_hot(labels, num_classes=self.cfg.dataset.num_classes).float()
        shuffled_labels_one_hot = F.one_hot(shuffled_labels, num_classes=self.cfg.dataset.num_classes).float()
        
        mixed_labels = lam * labels_one_hot + (1 - lam) * shuffled_labels_one_hot

        return mixed_images, mixed_labels

    def cutmix(self, images, labels, alpha):
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
        rand_index = torch.randperm(images.size(0))
        shuffled_images = images[rand_index]
        shuffled_labels = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        labels_one_hot = F.one_hot(labels, num_classes=self.cfg.dataset.num_classes).float()
        shuffled_labels_one_hot = F.one_hot(shuffled_labels, num_classes=self.cfg.dataset.num_classes).float()

        mixed_labels = lam * labels_one_hot + (1 - lam) * shuffled_labels_one_hot
        return images, mixed_labels

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = torch.clamp(torch.tensor(cx - cut_w // 2), 0, W).item()
        bby1 = torch.clamp(torch.tensor(cy - cut_h // 2), 0, H).item()
        bbx2 = torch.clamp(torch.tensor(cx + cut_w // 2), 0, W).item()
        bby2 = torch.clamp(torch.tensor(cy + cut_h // 2), 0, H).item()

        return bbx1, bby1, bbx2, bby2

    # --- Dataloader Hooks ---
    def train_dataloader(self):
        shuffle = True
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            # collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None,
            collate_fn=self.collate_fn if self.collate_fn is not None else None, # for mixup/cutmix augmentation
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def holdout_dataloader(self):
        return DataLoader(
            self.holdout_subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def _get_initial_sample_count(self, full_dataset):
        if "num_initial_labeled_real_samples" in self.cfg:
            return self.cfg.num_initial_labeled_real_samples
        elif "initial_labeled_dataset_ratio" in self.cfg:
            ratio = self.cfg.initial_labeled_dataset_ratio
            return int(len(full_dataset) * ratio)
        else:
            return len(full_dataset)

    def _load_initial_dataset(self, root, train_mode, transform):
        # Relative dataset paths are resolved against the repository root.
        root = resolve_project_path(root)
        if self.cfg.dataset.name == 'cifar10':
            return torchvision.datasets.CIFAR10(root=root, train=train_mode, transform=transform, download=True)
        elif self.cfg.dataset.name == 'mnist':
            return torchvision.datasets.MNIST(root=root, train=train_mode, transform=transform, download=True)
        elif self.cfg.dataset.name == 'imagenet_100':
            if train_mode:
                imagenet100_dataset = datasets.load_dataset("clane9/imagenet-100", split="train")
            else:
                imagenet100_dataset = datasets.load_dataset("clane9/imagenet-100", split="validation")
            imagenet100_dataset = imagenet100_dataset.cast_column("image", HFImage())  # PIL 지연 디코딩

            tfm = transform

            imagenet100_dataset = imagenet100_dataset.with_transform(ApplyTransformBatch(tfm))  # 배치 기준 transform
            return HFAsTorchDataset(imagenet100_dataset)  # (image_tensor, label) 튜플로 맞춤


        elif self.cfg.dataset.name == "imagenette":
            if train_mode:
                imagenette_dataset = torchvision.datasets.Imagenette(root=root, split='train', transform=transform, download=True)
            else:
                imagenette_dataset = torchvision.datasets.Imagenette(root=root, split='val', transform=transform, download=True)
            return imagenette_dataset
        
        elif self.cfg.dataset.name == "imagenet-100":
            if train_mode:
                imagenet100_dataset = datasets.load_dataset("clane9/imagenet-100", split="train")
            else:
                imagenet100_dataset = datasets.load_dataset("clane9/imagenet-100", split="validation")
            def transform_fn(example):
                example["image"] = transform(example["image"])
                example["label"] = example["label"]
                # return example
                return (example["image"], example["label"])
            return imagenet100_dataset.with_transform(transform_fn, columns=["image", "label"])
            
        else:
            raise ValueError(f"Dataset {self.cfg.dataset.dataset_name} is not supported.")

    # def _create_transform(self, train_mode):
    def _create_transform(self, transform_mode):
        assert transform_mode in ["train", "test", "inference"], f"transform_mode must be one of [train, test, inference], but got {transform_mode}" 
        # Random horizontal flip, random
        # Define any data augmentation transformations here
        transform_list = []
        # transform_list.append(torchvision.transforms.ToTensor())
        
        if transform_mode == "train":
            transform_cfg = self.cfg.dataset.transform.train
        elif transform_mode == "test":
            transform_cfg = self.cfg.dataset.transform.test
        elif transform_mode == "inference":
            transform_cfg = self.cfg.dataset.transform.inference
        
        compose_class = torchvision.transforms.Compose
        
        for transform_elem_cfg in transform_cfg:
            if self.cfg.target_model.task == "classification":
                transform = self.inflate_classification_transform(transform_elem_cfg)
            else:
                raise ValueError("Only the classification task is supported.")
            if transform is not None:
                transform_list.append(transform)
        compose_transform = compose_class(transform_list)

        return compose_transform


    def inflate_classification_transform(self, transform_elem_cfg):
        if "random_horizontal_flip" in transform_elem_cfg:
            return torchvision.transforms.RandomHorizontalFlip(
                p=transform_elem_cfg.random_horizontal_flip.p if 'p' in transform_elem_cfg.random_horizontal_flip else 0.5
            )
        elif "totensor" in transform_elem_cfg:
            return torchvision.transforms.ToTensor()
        elif "convert_to_rgb" in transform_elem_cfg:
            # return torchvision.transforms.Lambda(lambda img: img.convert("RGB"))
            return torchvision.transforms.Lambda(convert_rgb)
        elif "random_crop" in transform_elem_cfg:
            random_crop_cfg = transform_elem_cfg.random_crop
            return torchvision.transforms.RandomCrop(
                random_crop_cfg.size, 
                padding=random_crop_cfg.padding if 'padding' in random_crop_cfg else 0)
        
        elif "resize" in transform_elem_cfg:
            resize_cfg = transform_elem_cfg.resize

            interpolation_mode = torchvision.transforms.InterpolationMode.BILINEAR
            if 'interpolation' in resize_cfg:
                if resize_cfg.interpolation == "bilinear":
                    interpolation_mode = torchvision.transforms.InterpolationMode.BILINEAR
                else:
                    raise ValueError(f"Interpolation mode {resize_cfg.interpolation} is not supported.")
            return torchvision.transforms.Resize(
                size=resize_cfg.size if 'size' in resize_cfg else (32, 32),
                interpolation=interpolation_mode
            )
        elif "center_crop" in transform_elem_cfg:
            center_crop_cfg = transform_elem_cfg.center_crop
            return torchvision.transforms.CenterCrop(
                size=center_crop_cfg.size if 'size' in center_crop_cfg else (32, 32)
            )
        elif "random_resized_crop" in transform_elem_cfg:
            random_resized_crop_cfg = transform_elem_cfg.random_resized_crop
            return torchvision.transforms.RandomResizedCrop(
                size=random_resized_crop_cfg.size if 'size' in random_resized_crop_cfg else (32, 32),
                scale=random_resized_crop_cfg.scale if 'scale' in random_resized_crop_cfg else (0.08, 1.0),
                ratio=random_resized_crop_cfg.ratio if 'ratio' in random_resized_crop_cfg else (3. / 4., 4. / 3.)
            )
        elif "normalize" in transform_elem_cfg:
            normalize_cfg = transform_elem_cfg.normalize
            return torchvision.transforms.Normalize(
                mean=normalize_cfg.mean if 'mean' in normalize_cfg else (0.5, 0.5, 0.5),
                std=normalize_cfg.std if 'std' in normalize_cfg else (0.5, 0.5, 0.5)
            )
        else:
            print(f"Transform {transform_elem_cfg} is not supported.")
            return None
    

            

    @property
    def train_dataloader_len(self) -> int:
        """
        Returns the number of batches in the train dataloader.
        """
        if not self.train_dataset:
            # setup() has not been called yet
            return 0

        return len(self.train_dataset) // self.cfg.train.batch_size # Assume that drop_last is True


class DataModuleDeterministic(DataModule):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)

    # def _create_transform(self, train_mode):
    def _create_transform(self, transform_mode):
        # Random horizontal flip, random
        # Define any data augmentation transformations here

        # Deterministic transforms: use the `inference` transform if the dataset config defines one,
        # otherwise fall back to the `test` transform (classification datasets).
        mode = "inference" if "inference" in self.cfg.dataset.transform else "test"
        return super()._create_transform(transform_mode=mode)

    
    # --- Dataloader Hooks ---
    def train_dataloader(self):
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.cfg.train.num_workers,
        # )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def holdout_dataloader(self):
        return DataLoader(
            self.holdout_subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

class DataModuleDeterministicGeneratedDataForHoldout(DataModuleDeterministic):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)

    def set_holdout_dataset_from_train_dataset(self):
        if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
            sampling_range = set(range(len(self.train_dataset)))
            self.holdout_subset = torch.utils.data.Subset(
                self.train_dataset, random.sample(sampling_range, self.cfg.dataset.num_holdout_samples))
            print(f"Holdout dataset created: {len(self.holdout_subset)} samples", file=sys.stderr)

class DataModuleDeterministicClasswiseHoldout(DataModuleDeterministic):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)
        self.set_holdout_dataset_from_real_dataset()
    
    def set_holdout_dataset_from_real_dataset(self):
        num_classes = self.cfg.dataset.num_classes
        class_idx_dict = {i: [] for i in range(num_classes)}
        self.selected_idx_dict = {i: [] for i in range(num_classes)}
        if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
            # train_transforms = self._create_transform(train_mode=True)
            train_transforms = self._create_transform(transform_mode="train")
            full_train_dataset = self._load_initial_dataset(
                root=self.cfg.dataset.dataset_path, train_mode=True, transform=train_transforms)


            for idx, data in tqdm(enumerate(full_train_dataset), desc="Indexing classes for holdout dataset", total=len(full_train_dataset)):
                class_idx_dict[data[1]].append(idx)

            num_holdout_per_class = self.cfg.dataset.num_holdout_samples // num_classes
            for i in range(num_classes):
                self.selected_idx_dict[i] = random.sample(class_idx_dict[i], num_holdout_per_class)

            self.full_real_train_dataset = full_train_dataset
            self.holdout_subset = torch.utils.data.Subset(
                self.full_real_train_dataset, sum(self.selected_idx_dict.values(), []))
            print(f"Holdout dataset created: {len(self.holdout_subset)} samples", file=sys.stderr)

    def select_holdout_dataloader(self, class_idx):
        if not hasattr(self, 'selected_idx_dict'):
            raise ValueError("selected_idx_dict is not defined. Make sure to call set_holdout_dataset_from_train_dataset() first.")
        if class_idx not in self.selected_idx_dict:
            raise ValueError(f"Class index {class_idx} is not valid. Valid range is 0 to {len(self.selected_idx_dict)-1}.")
        
        selected_indices = self.selected_idx_dict[class_idx]
        subset = torch.utils.data.Subset(self.full_real_train_dataset, selected_indices)

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class DataModuleDeterministicInitialTrainingAsHoldout(DataModuleDeterministic):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)

    def setup(self, stage: str):
        super().setup(stage)
        if stage == 'fit' or stage is None:
            if not hasattr(self, "initial_labeled_real_subset"):
                raise ValueError("initial_labeled_real_subset is not defined. Make sure to set num_initial_labeled_real_samples > 0 in cfg.")
            self.set_holdout_dataset_from_initial_real_dataset()

    def set_holdout_dataset_from_initial_real_dataset(self):
        assert self.cfg.num_initial_labeled_real_samples > 0, "num_initial_labeled_real_samples must be greater than 0"

        num_classes = self.cfg.dataset.num_classes
        # class_idx_dict = {i: [] for i in range(num_classes)}
        self.selected_idx_dict = {i: [] for i in range(num_classes)}

        for idx, data in tqdm(enumerate(self.initial_labeled_real_subset), desc="Indexing classes for holdout dataset", total=len(self.initial_labeled_real_subset)):
            # class_idx_dict[data[1]].append(idx)
            self.selected_idx_dict[data[1]].append(idx)
        # num_holdout_per_class = self.cfg.dataset.num_holdout_samples // num_classes
        # for i in range(num_classes):
        #     breakpoint()
        #     self.selected_idx_dict[i] = random.sample(class_idx_dict[i], num_holdout_per_class)

        # self.full_real_train_dataset = full_train_dataset
        self.full_real_train_dataset = self.initial_labeled_real_subset
        self.holdout_subset = torch.utils.data.Subset(
            self.full_real_train_dataset, sum(self.selected_idx_dict.values(), []))
        print(f"Holdout dataset created: {len(self.holdout_subset)} samples", file=sys.stderr)
        self.holdout_subset = torch.utils.data.Subset(
            self.initial_labeled_real_subset, range(len(self.initial_labeled_real_subset))
        )

    def select_holdout_dataloader(self, class_idx):
        if not hasattr(self, 'selected_idx_dict'):
            raise ValueError("selected_idx_dict is not defined. Make sure to call set_holdout_dataset_from_train_dataset() first.")
        if class_idx not in self.selected_idx_dict:
            raise ValueError(f"Class index {class_idx} is not valid. Valid range is 0 to {len(self.selected_idx_dict)-1}.")
        
        selected_indices = self.selected_idx_dict[class_idx]
        subset = torch.utils.data.Subset(self.full_real_train_dataset, selected_indices)

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class DataModuleDeterministicGeneratedDataForHoldoutBalanced(DataModuleDeterministic):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)

    def set_holdout_dataset_from_train_dataset(self):
        if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
            num_class = self.cfg.dataset.num_classes
            class_idx_dict = {i: [] for i in range(num_class)}

            for idx, data in enumerate(self.train_dataset):
                label = data[1]
                class_idx_dict[label].append(idx)
            
            num_holdout_per_class = self.cfg.dataset.num_holdout_samples // num_class

            selected_indices = []
            for i in range(num_class):
                selected_indices += random.sample(class_idx_dict[i], num_holdout_per_class)
            
            self.holdout_subset = torch.utils.data.Subset(
                self.train_dataset, selected_indices)
            print(f"Holdout dataset created from train dataset: {len(self.holdout_subset)} samples", file=sys.stderr)

class DataModuleDeterministicClasswiseGeneratedDataForHoldoutBalanced(DataModuleDeterministicGeneratedDataForHoldoutBalanced):
    def __init__(self, cfg, num_processes=1, seed=42, cycle=0):
        super().__init__(cfg, num_processes, seed)
        self.cycle = cycle
    
    def set_holdout_dataset_from_train_dataset(self):
        seed = self.seed + self.cycle * 100
        random.seed(seed)
        # if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
        if self.cfg.dataset.get("num_holdout_samples", 0) > 0:
            num_class = self.cfg.dataset.num_classes
            class_idx_dict = {i: [] for i in range(num_class)}
            self.selected_idx_dict = {i: [] for i in range(num_class)}

            for idx, data in enumerate(self.train_dataset):
                label = data[1]
                class_idx_dict[label].append(idx)
            
            num_holdout_per_class = self.cfg.dataset.num_holdout_samples // num_class

            selected_indices = []
            for i in range(num_class):
                # selected_indices += random.sample(class_idx_dict[i], num_holdout_per_class)
                selected_indices_per_class = random.sample(class_idx_dict[i], num_holdout_per_class)
                selected_indices += selected_indices_per_class
                self.selected_idx_dict[i] = selected_indices_per_class
            
            self.holdout_subset = torch.utils.data.Subset(
                self.train_dataset, selected_indices)
            print(f"Holdout dataset created from train dataset: {len(self.holdout_subset)} samples", file=sys.stderr)
        
    
    def select_holdout_dataloader(self, class_idx):
        if not hasattr(self, 'selected_idx_dict'):
            raise ValueError("selected_idx_dict is not defined. Make sure to call set_holdout_dataset_from_train_dataset() first.")
        if class_idx not in self.selected_idx_dict:
            raise ValueError(f"Class index {class_idx} is not valid. Valid range is 0 to {len(self.selected_idx_dict)-1}.")
        
        selected_indices = self.selected_idx_dict[class_idx]
        subset = torch.utils.data.Subset(self.train_dataset, selected_indices)

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

class DataModuleDeterministicGeneratedDataSelectedByCLIPScore(DataModuleDeterministicGeneratedDataForHoldoutBalanced):
    def __init__(self, cfg, num_processes=1, seed=42, cycle=0, accelerator=None):
        super().__init__(cfg, num_processes, seed)
        from transformers import CLIPProcessor, CLIPModel
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = accelerator.device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.class_label_to_clip_text = self.get_formatted_texts_for_all_classes()
        self.cycle = cycle

    
    def get_formatted_texts_for_all_classes(self):
        prefix = "A photo of a"
        class_idx_to_name = get_class_idx_to_class_name_mapping(self.cfg)
        texts = []
        for i in range(self.cfg.dataset.num_classes):
            class_names = class_idx_to_name[i]
            if isinstance(class_names, list) or isinstance(class_names, tuple):
                # class_name = class_names[0]  # 첫 번째 이름 사용
                class_name = ", ".join(class_names)  # 모든 이름을 쉼표로 구분하여 사용
            else:
                class_name = class_names
            texts.append(f"{prefix} {class_name}")
        return texts
    
    
    def set_holdout_dataset_from_train_dataset(self):
        seed = self.seed + self.cycle * 100
        random.seed(seed)
        
        num_holdout_samples = self.cfg.dataset.get("num_holdout_samples", 0)
        if num_holdout_samples <= 0:
            return

        num_class = self.cfg.dataset.num_classes
        class_scores_dict = {i: [] for i in range(num_class)}
        
        # Use a reasonable batch size for inference
        # Adjust this based on your TPU memory
        batch_size = 256 

        # ✅ SOLUTION 1: Use a DataLoader for batching
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers, # Use multiple workers for data loading
            shuffle=False # No need to shuffle for this task
        )

        print("Pre-computing text features...", file=sys.stderr)
        # ✅ SOLUTION 2: Pre-compute text features once
        with torch.no_grad():
            # Get all class texts
            all_class_texts = self.class_label_to_clip_text 
            # Process them all in one batch
            text_inputs = self.clip_processor(text=all_class_texts, return_tensors="pt", padding=True).to(self.device)
            # Get the text features (embeddings)
            # Bypass get_text_features to be robust across transformers versions.
            text_outputs = self.clip_model.text_model(**text_inputs)
            text_features = self.clip_model.text_projection(text_outputs.pooler_output)
            # Normalize features for cosine similarity calculation
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print("Calculating CLIP scores for train dataset...", file=sys.stderr)
        
        # Store the original dataset indices for each sample
        original_indices = self.train_dataset.indices if isinstance(self.train_dataset, torch.utils.data.Subset) else list(range(len(self.train_dataset)))
        
        current_idx = 0
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Scoring dataset"):
                image_tensors, labels = batch_data
                
                # Move images to the TPU
                pixel_values = image_tensors.to(self.device)
                
                # Get image features from the model
                # Bypass get_image_features for the same reason as the text side.
                vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
                image_features = self.clip_model.visual_projection(vision_outputs.pooler_output)
                # Normalize image features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity (logits) between all images in the batch and all pre-computed text features
                # The result is a matrix of size [batch_size, num_classes]
                logits_per_image = image_features @ text_features.T
                
                # Get the specific score for the correct class for each image in the batch
                # We use `gather` to select the logit corresponding to the ground truth label
                scores = logits_per_image.gather(1, labels.to(self.device).view(-1, 1)).squeeze().cpu().tolist()
                
                # Handle the case where the batch size is 1
                if not isinstance(scores, list):
                    scores = [scores]

                # Store the (original_index, score) tuple for each sample in the batch
                for i, score in enumerate(scores):
                    label = labels[i].item()
                    # Get the original dataset index for this item
                    original_dataset_idx = original_indices[current_idx + i]
                    class_scores_dict[label].append((original_dataset_idx, score))
                
                current_idx += len(labels)
                

        # The rest of your logic for selecting samples remains the same...
        print("Selecting holdout samples based on CLIP scores...", file=sys.stderr)
        num_holdout_per_class = num_holdout_samples // num_class
        selected_indices = []
        self.selected_idx_dict = {i: [] for i in range(num_class)}

        for i in range(num_class):
            sorted_samples = sorted(class_scores_dict[i], key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in sorted_samples[:num_holdout_per_class]]
            selected_indices.extend(top_indices)
            self.selected_idx_dict[i] = top_indices
            
        self.holdout_subset = torch.utils.data.Subset(self.train_dataset, selected_indices)
        print(f"Holdout dataset created: {len(self.holdout_subset)} samples selected.", file=sys.stderr)

        # Clean up memory
        del self.clip_model, self.clip_processor, text_features
        torch.cuda.empty_cache() # Use torch.cuda.empty_cache() if on GPU, not needed for TPU

class DataModuleDeterministicClasswiseGeneratedDataSelectedByCLIPScore(DataModuleDeterministicClasswiseGeneratedDataForHoldoutBalanced):
    def __init__(self, cfg, num_processes=1, seed=42, cycle=0, accelerator=None):
        super().__init__(cfg, num_processes, seed, cycle)
        from transformers import CLIPProcessor, CLIPModel
        assert accelerator is not None, "accelerator must be provided"
        # self.device = torch.device(accelerator if torch.cuda.is_available() else "cpu")
        self.device = accelerator.device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.cycle = cycle
        self.class_label_to_clip_text = self.get_formatted_texts_for_all_classes()
    
    def get_formatted_texts_for_all_classes(self):
        prefix = "A photo of a"
        class_idx_to_name = get_class_idx_to_class_name_mapping(self.cfg)
        texts = []
        for i in range(self.cfg.dataset.num_classes):
            class_names = class_idx_to_name[i]
            if isinstance(class_names, list) or isinstance(class_names, tuple):
                # class_name = class_names[0]  # 첫 번째 이름 사용
                class_name = ", ".join(class_names)  # 모든 이름을 쉼표로 구분하여 사용
            else:
                class_name = class_names
            texts.append(f"{prefix} {class_name}")
        return texts
        
    def select_holdout_dataloader(self, class_idx):
        if not hasattr(self, 'selected_idx_dict'):
            raise ValueError("selected_idx_dict is not defined. Make sure to call set_holdout_dataset_from_train_dataset() first.")
        if class_idx not in self.selected_idx_dict:
            raise ValueError(f"Class index {class_idx} is not valid. Valid range is 0 to {len(self.selected_idx_dict)-1}.")
        
        selected_indices = self.selected_idx_dict[class_idx]
        subset = torch.utils.data.Subset(self.train_dataset, selected_indices)

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

### TMP
class DataModuleDeterministicClasswiseHoldoutUsingTestDataset(DataModuleDeterministic):
    def __init__(self, cfg, num_processes=1, seed=42):
        super().__init__(cfg, num_processes, seed)
        # self.set_holdout_dataset_from_real_dataset()
        self.set_holdout_dataset_from_real_test_dataset()

    def set_holdout_dataset_from_real_test_dataset(self):
        num_classes = self.cfg.dataset.num_classes
        class_idx_dict = {i: [] for i in range(num_classes)}
        self.selected_idx_dict = {i: [] for i in range(num_classes)}
        if self.cfg.dataset.get("use_train_data_as_holdout", False) and self.cfg.dataset.get("num_holdout_samples", 0) > 0:
            # train_transforms = self._create_transform(train_mode=True)
            train_transforms = self._create_transform(transform_mode="train")
            # full_train_dataset = self._load_initial_dataset(
            #     root=self.cfg.dataset.dataset_path, train_mode=True, transform=train_transforms)
            full_test_dataset = self._load_initial_dataset(
                root=self.cfg.dataset.dataset_path, train_mode=False, transform=train_transforms)

            for idx, data in enumerate(full_test_dataset):
                class_idx_dict[data[1]].append(idx)

            num_holdout_per_class = self.cfg.dataset.num_holdout_samples // num_classes
            for i in range(num_classes):
                self.selected_idx_dict[i] = random.sample(class_idx_dict[i], num_holdout_per_class)

            self.full_real_train_dataset = full_test_dataset
            self.holdout_subset = torch.utils.data.Subset(
                self.full_real_train_dataset, sum(self.selected_idx_dict.values(), []))
            print(f"DataModuleDeterministicClasswiseHoldoutUsingTestDataset: Holdout dataset created from test set: {len(self.holdout_subset)} samples", file=sys.stderr)
