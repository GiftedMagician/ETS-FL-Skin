import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os
from collections import defaultdict
import matplotlib.pyplot as plt


class NonIIDCIFARDataset(Dataset):
    """
    基于CIFAR数据集创建Non-IID分布的自定义数据集

    参数:
        base_dataset: CIFAR-10/100数据集对象
        indices: 该客户端包含的样本索引列表
        client_id: 客户端ID
        transform: 数据预处理变换
    """

    def __init__(self, base_dataset, indices, client_id, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.client_id = client_id
        self.transform = transform

        # 收集数据统计信息
        self.class_distribution = defaultdict(int)
        for idx in indices:
            _, label = base_dataset[idx]
            self.class_distribution[label] += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.base_dataset[actual_idx]

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """获取该客户端的类别分布信息"""
        return dict(self.class_distribution)

    def print_stats(self):
        """打印客户端数据统计信息"""
        total = len(self.indices)
        print(f"Client {self.client_id}: {total} samples")
        for cls, count in self.class_distribution.items():
            print(f"  Class {cls}: {count} samples ({count / total * 100:.1f}%)")


class FederatedDataLoader:
    """
    管理联邦学习数据分发

    参数:
        dataset_name: 数据集名称 ("CIFAR10" 或 "CIFAR100")
        num_clients: 客户端数量
        alpha: Dirichlet分布参数，控制数据异质性 (值越小，异质性越强)
        iid: 是否创建IID分布 (默认为False，使用Non-IID)
        data_root: 数据集存储路径
        seed: 随机种子
        val_ratio: 验证集比例 (每个客户端划分一部分作为本地验证集)
        train_batch_size: 训练集批大小
        val_batch_size: 验证集批大小
    """

    def __init__(self, dataset_name="CIFAR10", num_clients=100, alpha=0.3, iid=False,
                 data_root='./data', seed=42, val_ratio=0.1,
                 train_batch_size=32, val_batch_size=64):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.alpha = alpha
        self.iid = iid
        self.data_root = data_root
        self.seed = seed
        self.val_ratio = val_ratio
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 数据预处理
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 加载数据集
        self._load_datasets()

        # 划分数据
        self._partition_data()

    def _load_datasets(self):
        """加载CIFAR数据集"""
        if self.dataset_name == "CIFAR10":
            self.num_classes = 10
            train_dataset = datasets.CIFAR10(
                root=self.data_root, train=True, download=True, transform=None)
            test_dataset = datasets.CIFAR10(
                root=self.data_root, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == "CIFAR100":
            self.num_classes = 100
            train_dataset = datasets.CIFAR100(
                root=self.data_root, train=True, download=True, transform=None)
            test_dataset = datasets.CIFAR100(
                root=self.data_root, train=False, download=True, transform=self.transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=2)

        print(f"Loaded {self.dataset_name} dataset:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Number of classes: {self.num_classes}")

    def _partition_data(self):
        """划分数据到各个客户端"""
        # 获取所有训练样本的标签
        labels = np.array([label for _, label in self.train_dataset])

        if self.iid:
            # IID划分 - 随机均匀分配
            idxs = np.arange(len(self.train_dataset))
            np.random.shuffle(idxs)
            self.client_indices = np.array_split(idxs, self.num_clients)
        else:
            # Non-IID划分 - 基于Dirichlet分布
            self.client_indices = [[] for _ in range(self.num_clients)]

            # 为每个类别生成分配比例
            for class_idx in range(self.num_classes):
                # 获取当前类别的所有样本索引
                class_idxs = np.where(labels == class_idx)[0]
                np.random.shuffle(class_idxs)

                # 生成Dirichlet分布的比例
                proportions = np.random.dirichlet(
                    np.repeat(self.alpha, self.num_clients))

                # 根据比例分配样本
                proportions = np.array(
                    [p * (len(idx_j) < len(self.train_dataset) / self.num_clients)
                     for p, idx_j in zip(proportions, self.client_indices)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]

                # 分配样本给各个客户端
                self.client_indices = [
                    idx_j + idx.tolist() for idx_j, idx in zip(
                        self.client_indices, np.split(class_idxs, proportions))
                ]

        # 确保每个客户端都有样本
        self.client_indices = [np.array(idx) for idx in self.client_indices]

        # 打印数据分布信息
        self._print_data_distribution()

    def _print_data_distribution(self):
        """打印数据分布统计信息"""
        print("\nData distribution across clients:")
        client_samples = [len(indices) for indices in self.client_indices]
        print(f"  Total samples: {sum(client_samples)}")
        print(f"  Min samples per client: {min(client_samples)}")
        print(f"  Max samples per client: {max(client_samples)}")
        print(f"  Avg samples per client: {sum(client_samples) / len(client_samples):.1f}")

        # 计算每个类别的样本数分布
        global_class_dist = defaultdict(int)
        for indices in self.client_indices:
            for idx in indices:
                _, label = self.train_dataset[idx]
                global_class_dist[label] += 1

        # 打印类别分布
        print("\nGlobal class distribution:")
        for class_idx in range(self.num_classes):
            count = global_class_dist.get(class_idx, 0)
            print(f"  Class {class_idx}: {count} samples")

    def get_client_dataloader(self, client_id):
        """
        获取指定客户端的数据加载器

        返回:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 (如果val_ratio>0)
        """
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"Invalid client_id: {client_id}. Must be in [0, {self.num_clients - 1}]")

        # 获取该客户端的样本索引
        indices = self.client_indices[client_id]
        np.random.shuffle(indices)

        # 划分训练集和验证集
        val_size = int(len(indices) * self.val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # 创建数据集
        train_dataset = NonIIDCIFARDataset(
            self.train_dataset, train_indices, client_id, self.transform_train)
        val_dataset = NonIIDCIFARDataset(
            self.train_dataset, val_indices, client_id, self.transform_test)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.train_batch_size,
            shuffle=True, num_workers=2, pin_memory=True)

        val_loader = DataLoader(
            val_dataset, batch_size=self.val_batch_size,
            shuffle=False, num_workers=2, pin_memory=True)

        # 打印客户端数据统计
        print(f"\nClient {client_id} data distribution:")
        train_dataset.print_stats()

        return train_loader, val_loader

    def get_test_loader(self):
        """获取测试集数据加载器"""
        return self.test_loader

    def visualize_client_distribution(self, num_clients_to_show=5):
        """
        可视化客户端数据分布
        参数:
            num_clients_to_show: 要显示的客户端数量
        """

        plt.figure(figsize=(15, 10))

        # 选择要显示的客户端
        clients_to_show = min(num_clients_to_show, self.num_clients)
        client_ids = np.random.choice(self.num_clients, clients_to_show, replace=False)

        # 创建子图
        fig, axes = plt.subplots(clients_to_show, 1, figsize=(12, 3 * clients_to_show))
        if clients_to_show == 1:
            axes = [axes]

        for i, client_id in enumerate(client_ids):
            # 获取客户端数据分布
            _, val_loader = self.get_client_dataloader(client_id)
            dataset = val_loader.dataset
            class_dist = dataset.get_class_distribution()

            # 准备绘图数据
            classes = list(range(self.num_classes))
            counts = [class_dist.get(cls, 0) for cls in classes]

            # 绘制柱状图
            ax = axes[i]
            ax.bar(classes, counts, color='skyblue')
            ax.set_title(f'Client {client_id} Class Distribution (Total: {len(dataset)} samples)')
            ax.set_xlabel('Class')
            ax.set_ylabel('Sample Count')

            # 仅显示部分刻度标签
            if self.num_classes > 20:
                step = max(1, self.num_classes // 10)
                ax.set_xticks(classes[::step])
            else:
                ax.set_xticks(classes)

        plt.tight_layout()
        plt.savefig(f"{self.dataset_name}_client_distributions_alpha{self.alpha}.png", dpi=300)
        plt.show()

    def save_partition_info(self, save_dir="./partition_info"):
        """保存数据划分信息到文件"""
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{self.dataset_name}_alpha{self.alpha}_clients{self.num_clients}.npz"

        # 保存索引信息
        client_indices_array = [np.array(idx) for idx in self.client_indices]
        np.savez(
            filename,
            client_indices=client_indices_array,
            dataset_name=self.dataset_name,
            alpha=self.alpha,
            num_clients=self.num_clients,
            seed=self.seed
        )
        print(f"Partition information saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 创建高度异质的Non-IID数据分布
    fed_loader = FederatedDataLoader(
        dataset_name="CIFAR10",
        num_clients=10,
        alpha=0.3,  # 低alpha值表示高度异质
        iid=False,
        val_ratio=0.2,
        seed=42
    )

    # 可视化前5个客户端的数据分布
    fed_loader.visualize_client_distribution(num_clients_to_show=5)

    # 获取特定客户端的数据加载器
    client_id = 3
    train_loader, val_loader = fed_loader.get_client_dataloader(client_id)

    # 保存划分信息
    fed_loader.save_partition_info()

    # 使用示例
    print(f"\nExample batch from Client {client_id}:")
    for images, labels in train_loader:
        print(f"  Batch size: {images.shape}")
        print(f"  Labels: {labels[:10].tolist()}...")
        break