# models/model_manager.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelManager:
    """模型管理器，用于加载和配置不同的模型架构"""

    def __init__(self):
        self.available_models = {
            "simple": {
                "tsp": SimpleTSP,
                "cep": SimpleCEP,
                "description": "简单CNN模型 (类似LeNet)，训练快，内存占用小",
                "config": {
                    "learning_rate": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "local_epochs": 3
                }
            },
            "resnet": {
                "tsp": ResNetTSP,
                "cep": ResNetCEP,
                "description": "ResNet-based模型，精度高，但训练慢，内存占用大",
                "config": {
                    "learning_rate": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0005,
                    "local_epochs": 1
                }
            }
        }

    def get_model_info(self, model_name):
        """获取指定模型的详细信息"""
        if model_name not in self.available_models:
            raise ValueError(f"未知模型: {model_name}. 可用模型: {list(self.available_models.keys())}")
        return self.available_models[model_name]

    def create_models(self, model_name, num_classes=10):
        """创建指定模型的TSP和CEP部分"""
        model_info = self.get_model_info(model_name)

        # 创建TSP模型
        tsp_model = model_info["tsp"]()

        # 创建CEP模型
        cep_model = model_info["cep"](num_classes=num_classes)

        return tsp_model, cep_model, model_info

    def get_model_config(self, model_name):
        """获取模型特定的训练配置"""
        model_info = self.get_model_info(model_name)
        return model_info["config"]

    def list_models(self):
        """列出所有可用模型及其描述"""
        result = []
        for name, info in self.available_models.items():
            result.append(f"{name}: {info['description']}")
        return result


class BasicBlock(nn.Module):
    """基础残差块，适用于CIFAR数据集"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCEP(nn.Module):
    """在CPU/GPU上运行的高效部分（CEP部分）"""

    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=100):
        """
        参数:
            block: 残差块类型
            num_blocks: 每个层的块数列表
            num_classes: 分类类别数 (CIFAR10=10, CIFAR100=100)
        """
        super(ResNetCEP, self).__init__()

        # 默认块配置 (适用于CIFAR)
        if num_blocks is None:
            num_blocks = [3, 3]  # 对于ResNet18风格的CEP部分

        self.in_planes = 128  # 输入通道数，与TSP部分的输出匹配

        # 残差层
        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[1], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        """创建残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        # 输入x是TSP部分的输出特征 (batch_size, 128, 8, 8)

        # 通过残差层
        x = self.layer3(x)  # 输出: (batch_size, 256, 4, 4)
        x = self.layer4(x)  # 输出: (batch_size, 512, 2, 2)

        # 全局平均池化
        x = self.avgpool(x)  # 输出: (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # 输出: (batch_size, 512)

        # 分类层
        x = self.fc(x)  # 输出: (batch_size, num_classes)
        return x

    def get_gradients(self):
        """获取当前梯度，用于回传到TEE"""
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def compute_delta(self, reference_state):
        """计算与参考状态之间的权重差异"""
        current_state = self.state_dict()
        delta = {}
        for name in current_state:
            if name in reference_state:
                delta[name] = current_state[name] - reference_state[name]
        return delta


# 定义TSP模型 (与外部相同的结构)
class ResNetTSP(nn.Module):
    """在TEE中运行的敏感部分（TSP部分）"""

    def __init__(self):
        super(ResNetTSP, self).__init__()

        # 输入层 (适用于CIFAR的32x32图像)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 第一残差层
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        # 第二残差层
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),  # 下采样
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        # 输入x: (batch_size, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # 输出: (batch_size, 64, 32, 32)
        x = self.layer1(x)  # 输出: (batch_size, 64, 32, 32)
        x = self.layer2(x)  # 输出: (batch_size, 128, 16, 16)
        return x


class SimpleTSP(nn.Module):
    """TEE内运行的敏感部分（特征提取）"""

    def __init__(self):
        super(SimpleTSP, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出: 6x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 输出: 16x5x5
        return x


class SimpleCEP(nn.Module):
    """TEE外运行的高效部分（分类器）"""

    def __init__(self, num_classes=10):
        super(SimpleCEP, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_gradients(self):
        """获取当前梯度，用于回传到TEE"""
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def compute_delta(self, reference_state):
        """计算与参考状态之间的权重差异"""
        current_state = self.state_dict()
        delta = {}
        for name in current_state:
            if name in reference_state:
                delta[name] = current_state[name] - reference_state[name]
        return delta