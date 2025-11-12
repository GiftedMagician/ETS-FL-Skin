# config.py

class Config:
    """统一的配置管理器"""

    def __init__(self):
        # 模型配置
        self.model_name = "resnet"  # "simple" 或 "resnet"
        self.dataset_name = "CIFAR10"  # "CIFAR10" 或 "CIFAR100"

        # 联邦学习配置
        self.num_clients = 10
        self.num_groups = 2  # 客户端组数量
        self.alpha = 0.3  # Non-IID程度
        self.rounds = 5  # 联邦学习轮次

        # 训练配置
        self.batch_size = 32
        self.local_epochs = 1  # 客户端本地训练轮次

        # TEE配置
        self.tee_memory = 128  # MB
        self.feature_protection_method = "gaussian"  # 特征保护方法
        self.feature_noise_sigma = 0.01  # 高斯噪声标准差

        # 路径配置
        self.manifest_dir = "./enclave"
        self.workspace_base = "./workspace"
        self.results_dir = "./results"
        self.data_dir = "./data"

        # 实验标识
        self.experiment_id = None  # 自动生成时间戳标识

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"配置中没有属性 '{key}'")

    def get_num_classes(self):
        """根据数据集获取类别数量"""
        return 10 if self.dataset_name == "CIFAR10" else 100

    def validate(self):
        """验证配置的有效性"""
        if self.model_name not in ["simple", "resnet"]:
            raise ValueError("模型名称必须是 'simple' 或 'resnet'")

        if self.dataset_name not in ["CIFAR10", "CIFAR100"]:
            raise ValueError("数据集名称必须是 'CIFAR10' 或 'CIFAR100'")

        if self.num_clients < self.num_groups:
            raise ValueError("客户端数量不能小于组数量")

    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "配置参数:\n"
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                config_str += f"  {attr}: {getattr(self, attr)}\n"
        return config_str


# 创建全局配置实例
config = Config()