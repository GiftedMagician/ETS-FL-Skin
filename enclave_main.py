#!/storage/miniconda3/envs/p3.9Torch0.23/bin/python3
"""
SGX Enclave 主程序 - 完整实现版本
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
import base64
import numpy as np


# 定义所有支持的模型（与主程序保持一致）
class SimpleTSP(nn.Module):
    """简单CNN模型 - TSP部分"""

    def __init__(self):
        super(SimpleTSP, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class BasicBlock(nn.Module):
    """基础残差块，适用于CIFAR数据集"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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


class ResNetTSP(nn.Module):
    """ResNet TSP部分"""

    def __init__(self):
        super(ResNetTSP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1)
        )

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# 全局状态 - 每个客户端独立的模型和优化器
models = {}
initial_states = {}
optimizers = {}
model_types = {}


def create_model(model_type):
    """根据模型类型创建模型实例"""
    if model_type == "simple":
        return SimpleTSP()
    elif model_type == "resnet":
        return ResNetTSP()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def init_model(model_type, client_id):
    """为特定客户端初始化模型"""
    global models, initial_states, optimizers, model_types

    try:
        if client_id in models:
            return "SUCCESS: Model already initialized for client"

        # 创建模型实例
        model = create_model(model_type)
        models[client_id] = model
        model_types[client_id] = model_type

        # 创建优化器
        optimizers[client_id] = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # 初始状态为空，等待加载权重
        initial_states[client_id] = None

        print(f"DEBUG: Initialized {model_type} model for client {client_id}", file=sys.stderr)
        return "SUCCESS: Model initialized for client"

    except Exception as e:
        return f"ERROR: Failed to initialize model for client {client_id}: {str(e)}"


def load_model_from_pth(model_file, client_id):
    """为特定客户端从pth文件加载模型权重"""
    global models, initial_states, optimizers

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"

        # 读取模型文件
        state_dict = torch.load(model_file)

        # 加载模型权重
        models[client_id].load_state_dict(state_dict)

        # 保存初始状态用于增量计算
        if initial_states[client_id] is None:
            initial_states[client_id] = state_dict.copy()

        # 重置优化器状态
        optimizers[client_id] = torch.optim.SGD(
            models[client_id].parameters(),
            lr=0.01,
            momentum=0.9
        )

        print(f"DEBUG: Loaded model with {len(state_dict)} parameters for client {client_id}", file=sys.stderr)

        # 清理临时文件
        # if os.path.exists(model_file):
        #     os.remove(model_file)

        return "SUCCESS: Model weights loaded from pth file"

    except Exception as e:
        # if os.path.exists(model_file):
        #     os.remove(model_file)
        return f"ERROR: Failed to load model from pth file: {str(e)}"


def forward_from_pth(input_file, client_id):
    """为特定客户端执行前向传播"""
    global models

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"

        # 读取输入文件
        input_tensor = torch.load(input_file)

        # 确保模型在评估模式
        models[client_id].eval()

        # 执行前向传播
        with torch.no_grad():
            output_tensor = models[client_id](input_tensor)

        # 保存结果到文件
        timestamp = int(time.time() * 1000)
        result_file = f"/tmp/sgx_model_transfer/output_{client_id}_{timestamp}.pth"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        torch.save(output_tensor, result_file)

        # 清理输入文件
        # if os.path.exists(input_file):
        #     os.remove(input_file)

        return result_file

    except Exception as e:
        # if 'input_file' in locals() and os.path.exists(input_file):
        #     os.remove(input_file)
        return f"ERROR: Forward pass failed: {str(e)}"


def backward_from_pth(grad_file, client_id):
    """为特定客户端执行后向传播"""
    global models, optimizers

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"

        # 读取梯度文件
        gradients_dict = torch.load(grad_file)

        # 设置模型为训练模式
        models[client_id].train()

        # 设置梯度
        for name, param in models[client_id].named_parameters():
            if name in gradients_dict:
                param.grad = gradients_dict[name]

        # 执行优化步骤
        optimizers[client_id].step()
        optimizers[client_id].zero_grad()

        print(f"DEBUG: Executed backward pass for client {client_id}", file=sys.stderr)

        # 清理临时文件
        # if os.path.exists(grad_file):
        #     os.remove(grad_file)

        return "SUCCESS: Backward pass completed"

    except Exception as e:
        # if 'grad_file' in locals() and os.path.exists(grad_file):
        #     os.remove(grad_file)
        return f"ERROR: Backward pass failed: {str(e)}"


def get_delta_from_pth(client_id):
    """为特定客户端获取模型增量"""
    global models, initial_states

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"
        if initial_states[client_id] is None:
            return "ERROR: Model not properly initialized for client"

        # 获取当前状态
        current_state = models[client_id].state_dict()
        initial_state = initial_states[client_id]

        # 计算增量（当前状态 - 初始状态）
        delta_state = {}
        for key in current_state:
            if key in initial_state:
                delta_state[key] = current_state[key] - initial_state[key]

        # 保存增量到文件
        timestamp = int(time.time() * 1000)
        delta_file = f"/tmp/sgx_model_transfer/delta_{client_id}_{timestamp}.pth"
        os.makedirs(os.path.dirname(delta_file), exist_ok=True)

        torch.save(delta_state, delta_file)

        return delta_file

    except Exception as e:
        return f"ERROR: Failed to get delta: {str(e)}"


def update_global_from_pth(global_state_file, client_id):
    """为特定客户端更新全局模型"""
    global models, initial_states, optimizers

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"

        # 读取全局状态文件
        global_state = torch.load(global_state_file)

        # 加载全局状态到模型
        models[client_id].load_state_dict(global_state)

        # 更新初始状态（用于下一次增量计算）
        initial_states[client_id] = global_state.copy()

        # 重置优化器状态
        optimizers[client_id] = torch.optim.SGD(
            models[client_id].parameters(),
            lr=0.01,
            momentum=0.9
        )

        print(f"DEBUG: Updated global model for client {client_id}", file=sys.stderr)

        # 清理临时文件
        # if os.path.exists(global_state_file):
        #     os.remove(global_state_file)

        return "SUCCESS: Global model updated from pth file"

    except Exception as e:
        # if os.path.exists(global_state_file):
        #     os.remove(global_state_file)
        return f"ERROR: Failed to update global model: {str(e)}"


def get_model_info(client_id):
    """获取模型信息"""
    global models, model_types

    try:
        if client_id not in models:
            return "ERROR: Model not initialized for client"

        model = models[client_id]
        model_type = model_types[client_id]

        info = {
            "model_type": model_type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_dict_keys": list(model.state_dict().keys())
        }

        return json.dumps(info)

    except Exception as e:
        return f"ERROR: Failed to get model info: {str(e)}"


def test(message):
    """测试通信"""
    return f"{message}"


def cleanup_client(client_id):
    """清理特定客户端的资源"""
    global models, initial_states, optimizers, model_types

    try:
        if client_id in models:
            del models[client_id]
        if client_id in initial_states:
            del initial_states[client_id]
        if client_id in optimizers:
            del optimizers[client_id]
        if client_id in model_types:
            del model_types[client_id]

        return f"SUCCESS: Cleaned up resources for client {client_id}"
    except Exception as e:
        return f"ERROR: Failed to cleanup client {client_id}: {str(e)}"


def main():
    print("SGX Enclave initialized (complete implementation)", file=sys.stderr, flush=True)

    # 创建临时目录
    os.makedirs("/tmp/sgx_model_transfer", exist_ok=True)

    try:
        while True:
            # 读取输入
            line = sys.stdin.readline().strip()
            if not line:
                time.sleep(0.1)
                continue

            if line == "exit":
                print("Enclave shutting down...", flush=True)
                break

            # 解析命令
            parts = line.split(' ', 2)
            func_name = parts[0] if parts else ""
            arg1 = parts[1] if len(parts) > 1 else ""
            arg2 = parts[2] if len(parts) > 2 else ""

            # 处理命令并获取结果
            if func_name == "init_model":
                result = init_model(arg1, int(arg2)) if arg1 and arg2 else "ERROR: Missing arguments"
            elif func_name == "load_model_from_pth":
                result = load_model_from_pth(arg1, int(arg2)) if arg1 and arg2 else "ERROR: Missing arguments"
            elif func_name == "forward_from_pth":
                result = forward_from_pth(arg1, int(arg2)) if arg1 and arg2 else "ERROR: Missing arguments"
            elif func_name == "backward_from_pth":
                result = backward_from_pth(arg1, int(arg2)) if arg1 and arg2 else "ERROR: Missing arguments"
            elif func_name == "get_delta_from_pth":
                result = get_delta_from_pth(int(arg1)) if arg1 else "ERROR: Missing client_id"
            elif func_name == "update_global_from_pth":
                result = update_global_from_pth(arg1, int(arg2)) if arg1 and arg2 else "ERROR: Missing arguments"
            elif func_name == "get_model_info":
                result = get_model_info(int(arg1)) if arg1 else "ERROR: Missing client_id"
            elif func_name == "cleanup_client":
                result = cleanup_client(int(arg1)) if arg1 else "ERROR: Missing client_id"
            elif func_name == "test":
                result = test(arg1)
            elif func_name == "shutdown":
                result = "SHUTDOWN"
                break
            else:
                result = "UNKNOWN_COMMAND"

            # 将结果打印到标准输出
            print(result, flush=True)

    except Exception as e:
        print(f"ERROR: {str(e)}", flush=True)
    finally:
        # 清理临时目录
        try:
            import shutil
            shutil.rmtree("/tmp/sgx_model_transfer", ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    main()