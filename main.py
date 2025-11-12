import subprocess
import signal
import threading
import queue
import json
import base64
import time
import seaborn as sns
import pandas as pd
import torchvision
import datetime
from fedetated_dataloader import *
import torch.nn as nn
import torch.optim as optim
import copy
from config import config
from models import ModelManager
import select
import glob


def setup_experiment():
    """设置实验环境"""
    # 生成实验ID
    if config.experiment_id is None:
        config.experiment_id = time.strftime("%Y%m%d_%H%M%S")

    # 验证配置
    config.validate()

    # 创建必要的目录
    os.makedirs(config.workspace_base, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    # 打印配置
    print(config)


class SGXEnclaveManager:
    """管理Gramine/SGX enclave的生命周期"""

    def __init__(self, enclave_path, sign_key=None, memory_size=128):
        self.enclave_path = enclave_path
        self.sign_key = sign_key
        self.memory_size = memory_size
        self.process = None
        self.communication_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()

    def start_enclave(self):
        """启动SGX enclave进程"""
        if self.process and self.process.poll() is None:
            return True  # 已经运行

        try:
            # 构建Gramine命令
            cmd = [
                "gramine-sgx",
                "./pytorch",
                self.enclave_path
            ]

            # 直接使用conda环境的Python解释器
            # conda_python = "/storage/miniconda3/envs/p3.9Torch0.23/bin/python3"
            conda_python = "/home/wangbaoquan/anaconda3/envs/ETSFLp3.8/bin/python3"

            # 构建命令
            cmd = [
                conda_python,
                self.enclave_path
            ]


            self.process = subprocess.Popen(
                " ".join(cmd),
                # cwd='/home/wbq/ETS_FL/pytorch/',
                cwd='/home/wangbaoquan/code/ETS_FL/pytorch/',
                # cwd='/home/wbq/ETS_FL/enclaveTest/',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                shell=True,
                bufsize=1,  # 行缓冲
                # universal_newlines=True,  # 文本模式
                # text=True,  # 使用文本模式
                # universal_newlines=True,  # 确保使用文本模式
                preexec_fn=os.setsid
            )

            # 启动通信线程
            self.running = True
            self.comm_thread = threading.Thread(target=self._communication_handler)
            self.comm_thread.daemon = True
            self.comm_thread.start()

            # 等待enclave初始化完成并检查进程状态
            time.sleep(3)

            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                print(f"Enclave process failed to start. Error: {stderr_output}")
                self.running = False
                return False

            # 测试通信
            message = "ping"
            test_result = self.execute_in_enclave("test", message)
            if test_result == message:
                print("Enclave started successfully")
                return True
            else:
                print("Enclave communication test failed")
                return False

        except Exception as e:
            print(f"Failed to start enclave: {e}")
            self.running = False
            return False

    def stop_enclave(self):
        """停止SGX enclave进程"""
        with self.lock:
            self.running = False
            if self.process and self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=5)
                except Exception as e:
                    print(f"Error stopping enclave: {e}")
                finally:
                    self.process = None

    def _communication_handler(self):
        """处理与enclave的通信"""
        while self.running:
            try:
                # 检查进程状态
                if self.process is None or self.process.poll() is not None:
                    self.running = False
                    break

                if not self.communication_queue.empty():
                    data = self.communication_queue.get()
                    # 编码为字节
                    # 发送数据到enclave
                    # print(f"Sending to enclave: {data}")
                    data_bytes = (data + "\n").encode('utf-8')
                    self.process.stdin.write(data_bytes)
                    self.process.stdin.flush()

                    # 读取响应（字节模式）
                    start_time = time.time()
                    response_received = False

                    while time.time() - start_time < 300:  # 120秒超时
                        # 使用select检查是否有数据可读
                        if select.select([self.process.stdout], [], [], 0.1)[0]:
                            line_bytes = self.process.stdout.readline()
                            if line_bytes:
                                line = line_bytes.decode('utf-8', errors='ignore').strip()
                                # print(f"Received from enclave: {line}")
                                self.result_queue.put(line)
                                response_received = True
                                break
                        time.sleep(0.01)

                    if not response_received:
                        print(f"Timeout waiting for enclave response, Sending to enclave: {data}")
                        self.result_queue.put("TIMEOUT")

            except BrokenPipeError:
                print("Broken pipe: enclave process may have terminated")
                self.running = False
                break
            except Exception as e:
                print(f"Enclave communication error: {e}")
                time.sleep(0.1)

    def execute_in_enclave(self, function_name, *args):
        """在enclave中执行函数"""
        if not self.running or self.process is None or self.process.poll() is not None:
            print("Enclave is not running")
            return None

        command = f"{function_name} {' '.join(map(str, args))}"
        # print(f"Sending command to enclave: {command}")

        self.communication_queue.put(command)

        # 等待结果
        start_time = time.time()
        while time.time() - start_time < 300:  # 60秒超时
            if not self.result_queue.empty():
                result = self.result_queue.get()
                # print(f"Received result from enclave: {result}")
                return result
            time.sleep(0.1)

        print(f"Timeout waiting for enclave result, Sending command to enclave: {command}")
        return None

    def __del__(self):
        self.stop_enclave()

# class SGXEnclaveManager:
#     """管理Gramine/SGX enclave的生命周期"""
#
#     def __init__(self, enclave_path, sign_key=None, memory_size=128):
#         """
#         enclave_path: Gramine manifest文件路径
#         sign_key: 签名密钥路径
#         memory_size: 分配给enclave的内存大小(MB)
#         """
#         self.enclave_path = enclave_path
#         self.sign_key = sign_key
#         self.memory_size = memory_size
#         self.process = None
#         self.communication_queue = queue.Queue()
#         self.result_queue = queue.Queue()
#         self.running = False
#
#     def start_enclave(self):
#         """启动SGX enclave进程"""
#         if self.process and self.process.poll() is None:
#             return  # 已经运行
#
#         # 构建Gramine命令
#         cmd = [
#             "gramine-direct",
#             "./pytorch",
#             self.enclave_path
#         ]
#
#         # self.process = subprocess.Popen(
#         #     " ".join(cmd),
#         #     cwd='/home/wbq/ETS_FL/pytorch/',
#         #     shell=True,
#         #     stdout=subprocess.PIPE,
#         #     stderr=subprocess.PIPE,
#         #     stdin=subprocess.PIPE,
#         #     preexec_fn=os.setsid
#         # )
#
#         # 构建完整的命令序列
#         full_command = """
#         source ~/.bashrc &&
#         conda activate p3.9Torch0.23 &&
#         python3 {}
#         """.format(self.enclave_path)
#
#         self.process = subprocess.Popen(
#             full_command,
#             cwd='/home/wbq/ETS_FL/pytorch/',
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             stdin=subprocess.PIPE,
#             preexec_fn=os.setsid,
#             executable='/bin/bash'  # 确保使用bash
#         )
#
#         # if self.sign_key:
#         #     cmd.insert(1, f"-s {self.sign_key}")
#
#         # 启动通信线程
#         self.running = True
#         self.comm_thread = threading.Thread(target=self._communication_handler)
#         self.comm_thread.daemon = True
#         self.comm_thread.start()
#
#         # 等待enclave初始化完成
#         time.sleep(2)
#
#         # self.process.stdin.flush()
#         # # 读取响应
#         # output = self.process.stdout.readline().decode().strip()
#         # print(f"output:{output}")
#
#     def stop_enclave(self):
#         """停止SGX enclave进程"""
#         if self.process:
#             os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
#             self.process = None
#         self.running = False
#
#     def _communication_handler(self):
#         """处理与enclave的通信"""
#         while self.running:
#             if not self.communication_queue.empty():
#                 data = self.communication_queue.get()
#                 try:
#                     # 发送数据到enclave
#                     self.process.stdin.write(data.encode() + b"\n")
#                     self.process.stdin.flush()
#
#                     # 读取响应
#                     output = self.process.stdout.readline().decode().strip()
#                     self.result_queue.put(output)
#                     return output
#                 except Exception as e:
#                     print(f"Enclave communication error: {e}")
#             time.sleep(0.1)
#
#     def execute_in_enclave(self, function_name, *args):
#         """在enclave中执行函数"""
#         command = f"{function_name} {' '.join(map(str, args))}"
#         print(f"{function_name} {' '.join(map(str, args))}")
#         # self.process.stdin.write(command.encode() + b"\n")
#         # self.process.stdin.flush()
#         # # 读取响应
#         # output = self.process.stdout.readline().decode().strip()
#         # print(f"output:{output}")
#         # return output
#
#         self.communication_queue.put(command)
#         # 等待结果
#         start_time = time.time()
#         while time.time() - start_time < 300:  # 30秒超时
#             if not self.result_queue.empty():
#                 return self.result_queue.get()
#             time.sleep(0.1)
#         return None
#
#     def __del__(self):
#         self.stop_enclave()


# class SecureModelOperator:
#     """在SGX enclave中执行安全模型操作"""
#
#     def __init__(self, model, enclave_manager):
#         self.enclave = enclave_manager
#         self.model = model
#         self.initial_state = self._serialize_model_state()
#
#         # 首先根据模型类型初始化enclave中的模型
#         if hasattr(model, '__class__') and 'SimpleTSP' in model.__class__.__name__:
#             model_type = "simple"
#         else:
#             model_type = "resnet"  # 或其他检测逻辑
#
#         result = self.enclave.execute_in_enclave("init_model", model_type)
#         print(f"init_model result: {result}")
#         # if result != "SUCCESS":
#         #     raise RuntimeError(f"Failed to initialize model type: {result}")
#
#         # 然后加载模型权重
#         if not self.load_initial_state():
#             raise RuntimeError("Failed to load initial model state")
#
#     def load_initial_state(self):
#         """在enclave中加载初始模型状态"""
#         result = self.enclave.execute_in_enclave("load_model", self.initial_state)
#         print(f"load_initial_state result: {result}")
#         return result == "Model weights loaded successfully"
#
#     def _serialize_model_state(self):
#         """序列化模型状态为JSON字符串"""
#         state_dict = self.model.state_dict()
#         serialized = {}
#         for k, v in state_dict.items():
#             # 将张量转换为base64编码的字符串
#             array = v.numpy()
#             dtype = str(array.dtype)
#             shape = list(array.shape)
#             data = base64.b64encode(array.tobytes()).decode('utf-8')
#             serialized[k] = {"dtype": dtype, "shape": shape, "data": data}
#         return json.dumps(serialized)
#
#     def _deserialize_model_state(self, state_json):
#         """从JSON字符串反序列化模型状态"""
#         state_dict = {}
#         serialized = json.loads(state_json)
#         for k, v in serialized.items():
#             dtype = np.dtype(v["dtype"])
#             shape = tuple(v["shape"])
#             data = base64.b64decode(v["data"])
#             array = np.frombuffer(data, dtype=dtype).reshape(shape)
#             state_dict[k] = torch.from_numpy(array.copy())
#         return state_dict
#
#     def secure_forward(self, input_data):
#         """在enclave中执行安全的前向传播"""
#         # 序列化输入数据
#         if isinstance(input_data, torch.Tensor):
#             input_data = input_data.numpy()
#         input_b64 = base64.b64encode(input_data.tobytes()).decode('utf-8')
#         input_info = json.dumps({
#             "dtype": str(input_data.dtype),
#             "shape": list(input_data.shape),
#             "data": input_b64
#         })
#
#         # 在enclave中执行前向传播
#         result_json = self.enclave.execute_in_enclave("forward", input_info)
#         if not result_json:
#             return None
#
#         # 解析结果
#         result_data = json.loads(result_json)
#         output_b64 = result_data["data"]
#         output_dtype = np.dtype(result_data["dtype"])
#         output_shape = tuple(result_data["shape"])
#
#         output_bytes = base64.b64decode(output_b64)
#         output_array = np.frombuffer(output_bytes, dtype=output_dtype).reshape(output_shape)
#         return torch.from_numpy(output_array)
#
#     def secure_backward(self, gradients):
#         """在enclave中执行安全的后向传播和权重更新"""
#         # 序列化梯度
#         grad_dict = {}
#         for name, param in self.model.named_parameters():
#             if param.grad is not None:
#                 grad_array = param.grad.numpy()
#                 grad_b64 = base64.b64encode(grad_array.tobytes()).decode('utf-8')
#                 grad_dict[name] = {
#                     "dtype": str(grad_array.dtype),
#                     "shape": list(grad_array.shape),
#                     "data": grad_b64
#                 }
#
#         grad_json = json.dumps(grad_dict)
#
#         # 在enclave中执行后向传播
#         result = self.enclave.execute_in_enclave("backward", grad_json)
#         return result == "SUCCESS"
#
#     def get_model_delta(self):
#         """获取模型增量（当前状态与初始状态的差异）"""
#         result_json = self.enclave.execute_in_enclave("get_delta")
#         if not result_json:
#             return None
#         return self._deserialize_model_state(result_json)
#
#     def update_global_model(self, global_state_json):
#         """更新enclave中的全局模型状态"""
#         result = self.enclave.execute_in_enclave("update_global", global_state_json)
#         return result == "SUCCESS"

class SecureModelOperator:
    """在SGX enclave中执行安全模型操作 - 客户端隔离版本"""

    def __init__(self, model, enclave_manager, temp_dir, client_id):
        self.enclave = enclave_manager
        self.model = model
        self.client_id = client_id
        self.temp_dir = temp_dir  # 使用客户端特定的临时目录
        os.makedirs(self.temp_dir, exist_ok=True)

        # 等待enclave完全启动
        time.sleep(2)

        # 首先根据模型类型初始化enclave中的模型
        if hasattr(model, '__class__') and 'SimpleTSP' in model.__class__.__name__:
            model_type = "simple"
        else:
            model_type = "resnet"

        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            result = self.enclave.execute_in_enclave("init_model", model_type, self.client_id)
            print(f"Client {self.client_id}: init_model attempt {attempt + 1} result: {result}")
            if result and "SUCCESS" in result:
                break
            time.sleep(2)
        else:
            print(f"Warning: Model initialization may have failed for client {self.client_id}")

    def load_initial_state(self):
        """通过pth文件在enclave中加载初始模型状态"""
        try:
            # 保存模型状态到pth文件
            model_file = self._save_model_state_to_pth()
            if not model_file:
                return False

            # 通知enclave加载模型文件（传递client_id用于隔离）
            result = self.enclave.execute_in_enclave("load_model_from_pth", model_file, self.client_id)
            print(f"Client {self.client_id}: load_initial_state result: {result}")

            return "SUCCESS" in result

        except Exception as e:
            print(f"Error loading initial state for client {self.client_id}: {e}")
            return False

    def _save_model_state_to_pth(self, suffix=""):
        """将模型状态保存到pth文件"""
        try:
            # 生成唯一文件名（包含client_id）
            timestamp = int(time.time() * 1000)
            filename = f"model_state_{self.client_id}_{timestamp}{suffix}.pth"
            filepath = os.path.join(self.temp_dir, filename)

            # 直接保存模型状态字典
            torch.save(self.model.state_dict(), filepath)

            print(
                f"Client {self.client_id}: Model state saved to: {filepath} (size: {os.path.getsize(filepath)} bytes)")
            return filepath

        except Exception as e:
            print(f"Error saving model state to pth file for client {self.client_id}: {e}")
            return None

    def secure_forward(self, input_data):
        """在enclave中执行安全的前向传播"""
        try:
            # 保存输入数据到pth文件（包含client_id）
            timestamp = int(time.time() * 1000)
            input_file = os.path.join(self.temp_dir, f"input_{self.client_id}_{timestamp}.pth")

            # 直接保存输入张量
            if isinstance(input_data, torch.Tensor):
                torch.save(input_data, input_file)
            else:
                # 如果是numpy数组，转换为tensor
                input_tensor = torch.from_numpy(input_data)
                torch.save(input_tensor, input_file)

            # 在enclave中执行前向传播（传递client_id）
            result_file = self.enclave.execute_in_enclave("forward_from_pth", input_file, self.client_id)
            if not result_file or result_file.startswith("ERROR"):
                return None

            # 读取结果文件
            result_tensor = torch.load(result_file)

            # 清理临时文件
            if os.path.exists(input_file):
                os.remove(input_file)
            if os.path.exists(result_file):
                os.remove(result_file)

            return result_tensor

        except Exception as e:
            print(f"Error in secure_forward for client {self.client_id}: {e}")
            # 清理临时文件
            if 'input_file' in locals() and os.path.exists(input_file):
                os.remove(input_file)
            if 'result_file' in locals() and result_file and os.path.exists(result_file):
                os.remove(result_file)
            return None

    def update_global_model_from_file(self, global_state_file):
        """通过pth文件更新enclave中的全局模型状态"""
        result = self.enclave.execute_in_enclave("update_global_from_pth", global_state_file, self.client_id)
        return "SUCCESS" in result

    def secure_backward_from_pth(self, gradients):
        """通过pth文件在enclave中执行安全的后向传播"""
        try:
            # 保存梯度到pth文件（包含client_id）
            timestamp = int(time.time() * 1000)
            grad_file = os.path.join(self.temp_dir, f"gradients_{self.client_id}_{timestamp}.pth")

            # 直接保存梯度字典
            torch.save(gradients, grad_file)

            # 在enclave中执行后向传播（传递client_id）
            result = self.enclave.execute_in_enclave("backward_from_pth", grad_file, self.client_id)

            # 清理临时文件
            if os.path.exists(grad_file):
                os.remove(grad_file)

            return "SUCCESS" in result

        except Exception as e:
            print(f"Error in secure_backward_from_pth for client {self.client_id}: {e}")
            if 'grad_file' in locals() and os.path.exists(grad_file):
                os.remove(grad_file)
            return False

    def get_delta_from_pth(self):
        """通过pth文件获取模型增量"""
        result_file = self.enclave.execute_in_enclave("get_delta_from_pth", self.client_id)
        if not result_file or result_file.startswith("ERROR"):
            return None

        try:
            # 读取增量文件
            delta_state_dict = torch.load(result_file)

            # 清理临时文件
            if os.path.exists(result_file):
                os.remove(result_file)

            return delta_state_dict

        except Exception as e:
            print(f"Error loading delta from pth for client {self.client_id}: {e}")
            if os.path.exists(result_file):
                os.remove(result_file)
            return None

    # 保持向后兼容的方法
    def secure_backward(self, gradients):
        """向后兼容的secure_backward方法"""
        return self.secure_backward_from_pth(gradients)

    def get_model_delta(self):
        """向后兼容的get_model_delta方法"""
        return self.get_delta_from_pth()

class FeatureProtector:
    """特征保护模块"""

    def __init__(self, method="gaussian", sigma=0.01, projection_dim=128):
        self.method = method
        self.sigma = sigma
        self.projection_dim = projection_dim
        self.projection_matrix = None

    def protect(self, features):
        """应用保护机制到中间特征"""
        if self.method == "gaussian":
            noise = torch.randn_like(features) * self.sigma
            return features + noise
        elif self.method == "projection":
            return self.random_projection(features)
        return features

    def random_projection(self, features):
        """随机投影保护"""
        if self.projection_matrix is None:
            # 初始化投影矩阵
            input_dim = features.size(1)
            self.projection_matrix = torch.randn(input_dim, self.projection_dim)

        # 应用投影
        return torch.matmul(features, self.projection_matrix)


def visualize_results(result_file, show=True, save_dir="figures"):
    """
    可视化实验结果

    参数:
        result_file (str): 结果文件路径
        show (bool): 是否显示图表
        save_dir (str): 图表保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载结果数据
    with open(result_file, 'r') as f:
        result_data = json.load(f)

    # 提取数据
    config = result_data["config"]
    acc_history = result_data["accuracy"]["values"]
    comm_cost = result_data["communication"]["per_round"]
    time_cost = result_data["time"]["per_round"]

    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 准备文件名前缀
    timestamp = os.path.basename(result_file).split('_')[0]
    model_name = config.get("model", "ResNet18")
    dataset_name = config.get("dataset", "CIFAR10")
    num_clients = config.get("num_clients", 100)
    prefix = f"{timestamp}_{model_name}_{dataset_name}_C{num_clients}"

    # 1. 准确率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(acc_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Model Accuracy over Rounds\n{model_name} on {dataset_name}")
    plt.grid(True)

    # 标记最高准确率
    max_acc = max(acc_history)
    max_round = acc_history.index(max_acc)
    plt.annotate(f'Max: {max_acc:.2f}%',
                 xy=(max_round, max_acc),
                 xytext=(max_round + 5, max_acc - 5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 保存图表
    acc_plot_file = os.path.join(save_dir, f"{prefix}_accuracy.png")
    plt.savefig(acc_plot_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # 2. 通信开销分析
    plt.figure(figsize=(10, 6))

    # 通信开销（MB）
    comm_cost_mb = [c / (1024 * 1024) for c in comm_cost]

    # 柱状图展示每轮通信开销
    plt.bar(range(len(comm_cost_mb)), comm_cost_mb, color='skyblue')
    plt.xlabel("Communication Round")
    plt.ylabel("Communication Cost (MB)")
    plt.title(f"Communication Cost per Round\nTotal: {sum(comm_cost) / (1024 * 1024):.2f} MB")
    plt.grid(True, axis='y')

    # 保存图表
    comm_plot_file = os.path.join(save_dir, f"{prefix}_communication.png")
    plt.savefig(comm_plot_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # 3. 时间开销分析
    plt.figure(figsize=(10, 6))

    # 时间开销（分钟）
    time_cost_min = [t / 60 for t in time_cost]

    # 柱状图展示每轮时间开销
    plt.bar(range(len(time_cost_min)), time_cost_min, color='salmon')
    plt.xlabel("Communication Round")
    plt.ylabel("Time Cost (Minutes)")
    plt.title(f"Time Cost per Round\nTotal: {sum(time_cost) / 60:.2f} Minutes")
    plt.grid(True, axis='y')

    # 保存图表
    time_plot_file = os.path.join(save_dir, f"{prefix}_time.png")
    plt.savefig(time_plot_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # 4. 综合对比图（准确率与通信开销）
    plt.figure(figsize=(12, 8))

    # 创建双轴
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 准确率曲线
    color = 'tab:blue'
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(acc_history, 'b-o', linewidth=2, markersize=6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    # 通信开销曲线
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Comm Cost (MB)', color=color)
    ax2.plot(comm_cost_mb, 'r--s', linewidth=2, markersize=6, label='Comm Cost')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和图例
    plt.title(f"Accuracy vs Communication Cost\n{model_name} on {dataset_name}")
    fig.tight_layout()

    # 保存图表
    combined_plot_file = os.path.join(save_dir, f"{prefix}_combined.png")
    plt.savefig(combined_plot_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # 5. 创建结果摘要表
    summary_data = {
        "Metric": ["Final Accuracy", "Max Accuracy", "Average Accuracy",
                   "Total Comm Cost", "Average Comm Cost per Round",
                   "Total Time", "Average Time per Round"],
        "Value": [
            f"{result_data['accuracy']['final']:.2f}%",
            f"{result_data['accuracy']['max']:.2f}%",
            f"{result_data['accuracy']['avg']:.2f}%",
            f"{result_data['communication']['total'] / (1024 * 1024):.2f} MB",
            f"{result_data['communication']['avg_per_round'] / (1024 * 1024):.2f} MB",
            f"{result_data['time']['total'] / 60:.2f} minutes",
            f"{result_data['time']['avg_per_round']:.2f} seconds"
        ]
    }

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(save_dir, f"{prefix}_summary.csv")
    df.to_csv(summary_file, index=False)

    # 打印摘要
    print("\nExperiment Summary:")
    print(df.to_string(index=False))

    # 返回所有生成的文件路径
    return {
        "accuracy_plot": acc_plot_file,
        "communication_plot": comm_plot_file,
        "time_plot": time_plot_file,
        "combined_plot": combined_plot_file,
        "summary_table": summary_file
    }

def save_results(acc_history, comm_cost, time_cost, config, additional_data=None):
    """
    保存实验结果到JSON文件

    参数:
        acc_history (list): 每轮的准确率历史
        comm_cost (list): 每轮的通信开销（字节）
        time_cost (list): 每轮的时间开销（秒）
        config (dict): 实验配置
        additional_data (dict): 其他需要保存的数据
    """
    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 生成文件名（包含时间戳和配置信息）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model", "ResNet18")
    dataset_name = config.get("dataset", "CIFAR10")
    num_clients = config.get("num_clients", 100)
    filename = f"{results_dir}/{timestamp}_{model_name}_{dataset_name}_C{num_clients}.json"

    # 准备结果数据
    result_data = {
        "config": config,
        "accuracy": {
            "values": acc_history,
            "final": acc_history[-1] if acc_history else 0,
            "max": max(acc_history) if acc_history else 0,
            "avg": np.mean(acc_history) if acc_history else 0
        },
        "communication": {
            "per_round": comm_cost,
            "total": sum(comm_cost),
            "avg_per_round": np.mean(comm_cost) if comm_cost else 0,
            "max_per_round": max(comm_cost) if comm_cost else 0
        },
        "time": {
            "per_round": time_cost,
            "total": sum(time_cost),
            "avg_per_round": np.mean(time_cost) if time_cost else 0,
            "max_per_round": max(time_cost) if time_cost else 0
        },
        "additional": additional_data or {}
    }

    # 保存到文件
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {filename}")
    return filename


def get_cifar_test_loader(dataset_name="CIFAR10", batch_size=256):
    """
    获取CIFAR-10或CIFAR-100的测试集数据加载器

    参数:
        dataset_name (str): "CIFAR10" 或 "CIFAR100"
        batch_size (int): 批处理大小

    返回:
        DataLoader: 测试集数据加载器
    """
    # 数据预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 选择数据集
    if dataset_name == "CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset_name == "CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 加载测试集
    testset = dataset_class(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"Loaded {dataset_name} test set with {len(testset)} samples")
    return test_loader

# class SGXClient:
#     """使用SGX保护的客户端实现"""
#
#     def __init__(self, client_id, model_tsp, model_cep, enclave_manager,
#                  feature_protector, local_epochs=1, learning_rate=0.01,
#                  momentum=0.9, weight_decay=0.0001):
#         """
#         初始化客户端
#         """
#         self.id = client_id
#         self.enclave_manager = enclave_manager
#         self.secure_tsp = SecureModelOperator(model_tsp, enclave_manager)
#         self.model_cep = model_cep
#         self.feature_protector = feature_protector
#         self.local_epochs = local_epochs
#
#         # 设置优化器
#         self.optimizer = optim.SGD(
#             self.model_cep.parameters(),
#             lr=learning_rate,
#             momentum=momentum,
#             weight_decay=weight_decay
#         )
#
#         self.criterion = nn.CrossEntropyLoss()
#         self.train_loader = None
#         self.val_loader = None
#
#     def set_dataloaders(self, train_loader,val_loader):
#         """设置客户端数据加载器"""
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#
#     def initialize(self):
#         """初始化enclave中的模型"""
#         return self.secure_tsp.load_initial_state()
#
#     # resnet
#     # def train(self, global_tsp_state, global_cep_state):
#     #     """执行本地训练"""
#     #     # 更新全局模型状态
#     #     if not self.secure_tsp.update_global_model(global_tsp_state):
#     #         return None, None, None
#     #
#     #     # 将全局状态加载到CEP模型
#     #     cep_state = self.secure_tsp._deserialize_model_state(global_cep_state)
#     #     self.model_cep.load_state_dict(cep_state)
#     #
#     #     # 训练循环
#     #     total_loss = 0
#     #     for data, target in self.dataloader[0]:
#     #         # 在SGX中执行TSP前向传播
#     #         features = self.secure_tsp.secure_forward(data)
#     #         if features is None:
#     #             continue
#     #
#     #         # 特征保护
#     #         protected_features = self.feature_protector.protect(features)
#     #
#     #         # 在非安全区域执行CEP前向传播
#     #         output = self.model_cep(protected_features)
#     #         loss = torch.nn.functional.cross_entropy(output, target)
#     #         total_loss += loss.item()
#     #
#     #         # 反向传播
#     #         loss.backward()
#     #
#     #         # 在SGX中执行TSP反向传播
#     #         if not self.secure_tsp.secure_backward(dict(self.model_cep.named_parameters())):
#     #             break
#     #
#     #         if not True:
#     #             break
#     #
#     #         # 清空CEP梯度
#     #         self.model_cep.zero_grad()
#     #
#     #     # 获取模型增量
#     #     delta_tsp = self.secure_tsp.get_model_delta()
#     #     if delta_tsp is None:
#     #         return None, None, total_loss
#     #
#     #     # 计算CEP增量
#     #     current_cep_state = self.model_cep.state_dict()
#     #     delta_cep = {}
#     #     for name in current_cep_state:
#     #         delta_cep[name] = current_cep_state[name] - cep_state[name]
#     #
#     #     return delta_tsp, delta_cep, total_loss
#
#     def train(self, global_tsp_state, global_cep_state):
#         """执行本地训练"""
#         # 更新全局模型状态
#         if not self.secure_tsp.update_global_model(global_tsp_state):
#             return None, None, None
#
#         # 将全局状态加载到CEP模型
#         cep_state = self.secure_tsp._deserialize_model_state(global_cep_state)
#         self.model_cep.load_state_dict(cep_state)
#
#         # 保存初始CEP状态用于增量计算
#         initial_cep_state = copy.deepcopy(self.model_cep.state_dict())
#
#         # 多轮本地训练
#         total_loss = 0
#         num_batches = len(self.train_loader)
#
#         for epoch in range(self.local_epochs):
#             epoch_loss = 0
#
#             for batch_idx, (data, target) in enumerate(self.train_loader):
#                 # 在SGX中执行TSP前向传播
#                 features = self.secure_tsp.secure_forward(data)
#                 if features is None:
#                     continue
#
#                 # 特征保护
#                 protected_features = self.feature_protector.protect(features)
#
#                 # 在非安全区域执行CEP前向传播
#                 output = self.model_cep(protected_features)
#                 loss = self.criterion(output, target)
#                 epoch_loss += loss.item()
#
#                 # 反向传播
#                 loss.backward()
#
#                 # 在SGX中执行TSP反向传播
#                 if not self.secure_tsp.secure_backward(self.model_cep.get_gradients()):
#                     break
#
#                 # 更新CEP权重
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#
#                 # 打印进度
#                 if batch_idx % 10 == 0:
#                     print(f"Client {self.id}: Epoch {epoch + 1}/{self.local_epochs}, "
#                           f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
#
#             total_loss += epoch_loss / num_batches
#
#         # 获取模型增量
#         delta_tsp = self.secure_tsp.get_model_delta()
#         if delta_tsp is None:
#             return None, None, total_loss / self.local_epochs
#
#         # 计算CEP增量
#         current_cep_state = self.model_cep.state_dict()
#         delta_cep = {}
#         for name in current_cep_state:
#             if name in initial_cep_state:
#                 delta_cep[name] = current_cep_state[name] - initial_cep_state[name]
#
#         return delta_tsp, delta_cep, total_loss / self.local_epochs

class SGXClient:
    """使用SGX保护的客户端实现 - 每个客户端独立"""

    def __init__(self, client_id, model_tsp, model_cep, enclave_manager,
                 feature_protector, local_epochs=1, learning_rate=0.01,
                 momentum=0.9, weight_decay=0.0001):
        """
        初始化客户端
        """
        self.id = client_id
        self.enclave_manager = enclave_manager
        self.model_tsp = model_tsp
        self.model_cep = model_cep
        self.feature_protector = feature_protector
        self.local_epochs = local_epochs

        # 每个客户端有自己的临时目录
        self.temp_dir = f"/tmp/sgx_client_{client_id}_transfer"
        os.makedirs(self.temp_dir, exist_ok=True)

        # 每个客户端有自己的 SecureModelOperator
        self.secure_tsp = SecureModelOperator(
            model_tsp, enclave_manager, self.temp_dir, client_id
        )

        # 设置优化器
        self.optimizer = optim.SGD(
            self.model_cep.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = None
        self.val_loader = None

    def set_dataloaders(self, train_loader, val_loader):
        """设置客户端数据加载器"""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def initialize(self):
        """初始化enclave中的模型"""
        return self.secure_tsp.load_initial_state()

    def train(self, global_tsp_state, global_cep_state):
        """执行本地训练"""
        try:
            # 保存全局状态到客户端的临时目录
            tsp_state_file = self._save_state_to_pth(global_tsp_state, "global_tsp")
            cep_state_file = self._save_state_to_pth(global_cep_state, "global_cep")

            if not tsp_state_file or not cep_state_file:
                return None, None, None

            # 更新enclave中的全局模型状态
            if not self.secure_tsp.update_global_model_from_file(tsp_state_file):
                return None, None, None

            # 将全局状态加载到CEP模型
            if isinstance(global_cep_state, str) and global_cep_state.endswith('.pth'):
                cep_state = torch.load(global_cep_state)
            else:
                cep_state = global_cep_state
            self.model_cep.load_state_dict(cep_state)

            # 保存初始CEP状态用于增量计算
            initial_cep_state = copy.deepcopy(self.model_cep.state_dict())

            # 多轮本地训练
            total_loss = 0
            num_batches = len(self.train_loader)

            for epoch in range(self.local_epochs):
                epoch_loss = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # 在SGX中执行TSP前向传播
                    features = self.secure_tsp.secure_forward(data)
                    if features is None:
                        continue

                    # 特征保护
                    protected_features = self.feature_protector.protect(features)

                    # 在非安全区域执行CEP前向传播
                    output = self.model_cep(protected_features)
                    loss = self.criterion(output, target)
                    epoch_loss += loss.item()

                    # 反向传播
                    loss.backward()

                    # 获取CEP模型的梯度
                    cep_gradients = {}
                    for name, param in self.model_cep.named_parameters():
                        if param.grad is not None:
                            cep_gradients[name] = param.grad

                    # 在SGX中执行TSP反向传播
                    if not self.secure_tsp.secure_backward_from_pth(cep_gradients):
                        break

                    # 更新CEP权重
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 打印进度
                    if batch_idx % 10 == 0:
                        print(f"Client {self.id}: Epoch {epoch + 1}/{self.local_epochs}, "
                              f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

                total_loss += epoch_loss / num_batches

            # 获取TSP模型增量
            delta_tsp = self.secure_tsp.get_delta_from_pth()
            if delta_tsp is None:
                return None, None, total_loss / self.local_epochs

            # 计算CEP增量
            current_cep_state = self.model_cep.state_dict()
            delta_cep = {}
            for name in current_cep_state:
                if name in initial_cep_state:
                    delta_cep[name] = current_cep_state[name] - initial_cep_state[name]

            # 清理临时文件
            self._cleanup_temp_files()

            return delta_tsp, delta_cep, total_loss / self.local_epochs

        except Exception as e:
            print(f"Error in client {self.id} training: {e}")
            self._cleanup_temp_files()
            return None, None, None

    def _save_state_to_pth(self, state_dict, prefix):
        """保存状态字典到pth文件"""
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{prefix}_{timestamp}.pth"
            filepath = os.path.join(self.temp_dir, filename)

            # 如果state_dict是字符串，可能是文件路径，直接返回
            if isinstance(state_dict, str) and state_dict.endswith('.pth'):
                return state_dict

            torch.save(state_dict, filepath)

            return filepath

        except Exception as e:
            print(f"Error saving state to pth file: {e}")
            return None

    def _cleanup_temp_files(self):
        """清理客户端的临时文件"""
        try:
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up temp files for client {self.id}: {e}")

    def cleanup(self):
        """清理客户端资源"""
        self._cleanup_temp_files()
        try:
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except:
            pass

    def __del__(self):
        self.cleanup()

# class FederatedServer:
#     """联邦学习服务器"""
#
#     def __init__(self, model_tsp, model_cep, num_groups=10):
#         self.global_tsp_model = model_tsp
#         self.global_cep_model = model_cep
#         self.groups = []
#         self.compression_ratio = []
#
#     def initialize_global_model(self):
#         """初始化全局模型权重"""
#         self.global_tsp_model
#
#     def create_groups(self, clients, group_size=10):
#         """创建客户端组"""
#         self.groups = []
#         for i in range(0, len(clients), group_size):
#             group_clients = clients[i:i + group_size]
#             self.groups.append(Group(i // group_size, group_clients))
#
#     def aggregate(self, group_deltas):
#         """聚合组更新"""
#         # 计算平均增量
#         avg_delta_tsp = {}
#         avg_delta_cep = {}
#
#         for delta_tsp, delta_cep in group_deltas:
#             for name in delta_tsp:
#                 if name not in avg_delta_tsp:
#                     avg_delta_tsp[name] = delta_tsp[name].clone()
#                 else:
#                     avg_delta_tsp[name] += delta_tsp[name]
#
#             for name in delta_cep:
#                 if name not in avg_delta_cep:
#                     avg_delta_cep[name] = delta_cep[name].clone()
#                 else:
#                     avg_delta_cep[name] += delta_cep[name]
#
#         num_groups = len(group_deltas)
#         for name in avg_delta_tsp:
#             avg_delta_tsp[name] /= num_groups
#         for name in avg_delta_cep:
#             avg_delta_cep[name] /= num_groups
#
#         # 更新全局模型
#         current_tsp_state = self.global_tsp_model.state_dict()
#         current_cep_state = self.global_cep_model.state_dict()
#
#         for name in avg_delta_tsp:
#             current_tsp_state[name] += avg_delta_tsp[name]
#
#         for name in avg_delta_cep:
#             current_cep_state[name] += avg_delta_cep[name]
#
#         self.global_tsp_model.load_state_dict(current_tsp_state)
#         self.global_cep_model.load_state_dict(current_cep_state)
#
#         return current_tsp_state, current_cep_state
#
#     def run_federated_learning(self, rounds, test_loader):
#         """执行联邦学习过程"""
#         acc_history = []
#         comm_cost = []
#         time_cost = []
#
#         # 获取初始全局状态
#         global_tsp_state = self.global_tsp_model.state_dict()
#         global_cep_state = self.global_cep_model.state_dict()
#
#         # 序列化全局状态
#         serialized_tsp = self._serialize_state(global_tsp_state)
#         serialized_cep = self._serialize_state(global_cep_state)
#
#         for r in range(rounds):
#             round_start = time.time()
#             group_deltas = []
#
#             # 选择组参与训练
#             for group in self.groups:
#                 # 执行组内串行训练
#                 delta_tsp, delta_cep, group_time = group.sequential_train(
#                     serialized_tsp, serialized_cep
#                 )
#
#                 if delta_tsp and delta_cep:
#                     group_deltas.append((delta_tsp, delta_cep))
#                     time_cost.append(group_time)
#
#             # 聚合更新
#             new_tsp_state, new_cep_state = self.aggregate(group_deltas)
#             serialized_tsp = self._serialize_state(new_tsp_state)
#             serialized_cep = self._serialize_state(new_cep_state)
#
#             # 评估模型
#             accuracy = self.evaluate(test_loader)
#             acc_history.append(accuracy)
#
#             # 计算通信成本
#             round_comm = sum(len(serialized_tsp) + len(serialized_cep) for _ in group_deltas)
#             comm_cost.append(round_comm)
#
#             print(f"Round {r + 1}/{rounds} - Accuracy: {accuracy:.2f}% - "
#                   f"Comm Cost: {round_comm / (1024 * 1024):.2f} MB - "
#                   f"Time: {time.time() - round_start:.2f}s")
#
#         return acc_history, comm_cost, time_cost
#
#     def _serialize_state(self, state_dict):
#         """序列化模型状态为JSON字符串"""
#         serialized = {}
#         for k, v in state_dict.items():
#             array = v.numpy()
#             dtype = str(array.dtype)
#             shape = list(array.shape)
#             data = base64.b64encode(array.tobytes()).decode('utf-8')
#             serialized[k] = {"dtype": dtype, "shape": shape, "data": data}
#         return json.dumps(serialized)
#
#     def evaluate(self, test_loader):
#         """评估全局模型性能"""
#         self.global_tsp_model.eval()
#         self.global_cep_model.eval()
#
#         correct = 0
#         total = 0
#
#         with torch.no_grad():
#             for data, target in test_loader:
#                 # 使用TSP模型提取特征
#                 features = self.global_tsp_model(data)
#
#                 # 使用CEP模型分类
#                 output = self.global_cep_model(features)
#                 _, predicted = torch.max(output.data, 1)
#
#                 total += target.size(0)
#                 correct += (predicted == target).sum().item()
#
#         return 100 * correct / total

class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, model_tsp, model_cep, num_groups=10):
        self.global_tsp_model = model_tsp
        self.global_cep_model = model_cep
        self.groups = []
        self.compression_ratio = []
        self.temp_dir = "/tmp/fed_server_transfer"
        os.makedirs(self.temp_dir, exist_ok=True)

    def initialize_global_model(self):
        """初始化全局模型权重"""
        # 这个方法可以留空，或者添加实际的初始化逻辑
        pass

    def create_groups(self, clients, group_size=10):
        """创建客户端组"""
        self.groups = []
        for i in range(0, len(clients), group_size):
            group_clients = clients[i:i + group_size]
            self.groups.append(Group(i // group_size, group_clients))

    def _save_state_to_pth(self, state_dict, prefix):
        """保存状态字典到pth文件"""
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.pth"
        filepath = os.path.join(self.temp_dir, filename)
        torch.save(state_dict, filepath)
        return filepath

    def aggregate(self, group_deltas):
        """聚合组更新 - pth文件版本"""
        try:
            avg_delta_tsp, avg_delta_cep = {}, {}

            for delta_tsp, delta_cep in group_deltas:
                # 加载文件
                if isinstance(delta_tsp, str) and delta_tsp.endswith('.pth'):
                    delta_tsp = torch.load(delta_tsp)
                if isinstance(delta_cep, str) and delta_cep.endswith('.pth'):
                    delta_cep = torch.load(delta_cep)

                # 聚合 tsp
                for name, param in delta_tsp.items():
                    if name not in avg_delta_tsp:
                        avg_delta_tsp[name] = param.detach().float().clone()
                    else:
                        avg_delta_tsp[name] += param.detach().float()

                # 聚合 cep
                for name, param in delta_cep.items():
                    if name not in avg_delta_cep:
                        avg_delta_cep[name] = param.detach().float().clone()
                    else:
                        avg_delta_cep[name] += param.detach().float()

            num_groups = len(group_deltas)

            for name in avg_delta_tsp:
                if avg_delta_tsp[name].dtype.is_floating_point:
                    avg_delta_tsp[name] /= num_groups

            for name in avg_delta_cep:
                if avg_delta_cep[name].dtype.is_floating_point:
                    avg_delta_cep[name] /= num_groups

            # 更新全局模型
            current_tsp_state = self.global_tsp_model.state_dict()
            current_cep_state = self.global_cep_model.state_dict()

            for name in avg_delta_tsp:
                if name in current_tsp_state:
                    current_tsp_state[name] += avg_delta_tsp[name].to(current_tsp_state[name].dtype)

            for name in avg_delta_cep:
                if name in current_cep_state:
                    current_cep_state[name] += avg_delta_cep[name].to(current_cep_state[name].dtype)

            self.global_tsp_model.load_state_dict(current_tsp_state)
            self.global_cep_model.load_state_dict(current_cep_state)

            tsp_file = self._save_state_to_pth(current_tsp_state, "global_tsp")
            cep_file = self._save_state_to_pth(current_cep_state, "global_cep")

            return tsp_file, cep_file

        except Exception as e:
            print(f"Error in aggregation: {e}")
            return None, None

    def run_federated_learning(self, rounds, clients, test_loader):
        """执行联邦学习过程 - 修正参数"""
        acc_history = []
        comm_cost = []
        time_cost = []

        # 创建客户端组
        group_size = max(1, len(clients) // 10)  # 默认每组10个客户端
        self.create_groups(clients, group_size)

        # 获取初始全局状态
        global_tsp_state = self.global_tsp_model.state_dict()
        global_cep_state = self.global_cep_model.state_dict()

        # 保存为pth文件
        serialized_tsp = self._save_state_to_pth(global_tsp_state, "global_tsp")
        serialized_cep = self._save_state_to_pth(global_cep_state, "global_cep")

        for r in range(rounds):
            print(f"Start round {r+1}")
            round_start = time.time()
            group_deltas = []

            # 选择组参与训练
            for group in self.groups:
                # 执行组内串行训练
                delta_tsp, delta_cep, group_time = group.sequential_train(
                    serialized_tsp, serialized_cep
                )

                if delta_tsp and delta_cep:
                    group_deltas.append((delta_tsp, delta_cep))
                    time_cost.append(group_time)

            # 聚合更新
            new_tsp_file, new_cep_file = self.aggregate(group_deltas)
            if new_tsp_file and new_cep_file:
                serialized_tsp = new_tsp_file
                serialized_cep = new_cep_file
            else:
                print(f"Round {r + 1}: Aggregation failed, using previous state")
                continue

            # 评估模型
            accuracy = self.evaluate(test_loader)
            acc_history.append(accuracy)

            # 计算通信成本（基于文件大小）
            tsp_size = os.path.getsize(serialized_tsp) if os.path.exists(serialized_tsp) else 0
            cep_size = os.path.getsize(serialized_cep) if os.path.exists(serialized_cep) else 0
            round_comm = (tsp_size + cep_size) * len(group_deltas)
            comm_cost.append(round_comm)

            print(f"Round {r + 1}/{rounds} - Accuracy: {accuracy:.2f}% - "
                  f"Comm Cost: {round_comm / (1024 * 1024):.2f} MB - "
                  f"Time: {time.time() - round_start:.2f}s")

        # 清理临时文件
        self._cleanup_temp_files()

        return acc_history, comm_cost, time_cost

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

    def evaluate(self, test_loader):
        """评估全局模型性能"""
        self.global_tsp_model.eval()
        self.global_cep_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                # 使用TSP模型提取特征
                features = self.global_tsp_model(data)

                # 使用CEP模型分类
                output = self.global_cep_model(features)
                _, predicted = torch.max(output.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100 * correct / total


# class Group:
#     """客户端组，管理串行训练"""
#
#     def __init__(self, group_id, clients):
#         self.id = group_id
#         self.clients = clients
#         self.leader = clients[0]  # 组长客户端
#
#     def sequential_train(self, global_tsp_state, global_cep_state):
#         """执行组内串行训练"""
#         start_time = time.time()
#         tsp_state = global_tsp_state
#         cep_state = global_cep_state
#         group_delta_tsp = None
#         group_delta_cep = None
#
#         # 客户端串行训练
#         for client in self.clients:
#             # 客户端训练
#             delta_tsp, delta_cep, loss = client.train(tsp_state, cep_state)
#
#             if delta_tsp is None or delta_cep is None:
#                 continue
#
#             # 更新模型状态
#             tsp_state = self._apply_delta(tsp_state, delta_tsp)
#             cep_state = self._apply_delta(cep_state, delta_cep)
#
#             # 累积增量
#             if group_delta_tsp is None:
#                 group_delta_tsp = delta_tsp
#                 group_delta_cep = delta_cep
#             else:
#                 for name in delta_tsp:
#                     group_delta_tsp[name] += delta_tsp[name]
#                 for name in delta_cep:
#                     group_delta_cep[name] += delta_cep[name]
#
#         # 平均增量
#         num_clients = len(self.clients)
#         if group_delta_tsp:
#             for name in group_delta_tsp:
#                 group_delta_tsp[name] /= num_clients
#         if group_delta_cep:
#             for name in group_delta_cep:
#                 group_delta_cep[name] /= num_clients
#
#         return group_delta_tsp, group_delta_cep, time.time() - start_time
#
#     def _apply_delta(self, serialized_state, delta):
#         """应用增量到序列化状态"""
#         state_dict = json.loads(serialized_state)
#         delta_dict = {}
#
#         # 将delta转换为可序列化格式
#         for name, tensor in delta.items():
#             array = tensor.numpy()
#             dtype = str(array.dtype)
#             shape = list(array.shape)
#             data = base64.b64encode(array.tobytes()).decode('utf-8')
#             delta_dict[name] = {"dtype": dtype, "shape": shape, "data": data}
#
#         # 应用增量
#         for name in state_dict:
#             if name in delta_dict:
#                 # 解码原始数据
#                 orig_data = base64.b64decode(state_dict[name]["data"])
#                 orig_array = np.frombuffer(
#                     orig_data,
#                     dtype=np.dtype(state_dict[name]["dtype"])
#                 ).reshape(tuple(state_dict[name]["shape"]))
#
#                 # 解码增量
#                 delta_data = base64.b64decode(delta_dict[name]["data"])
#                 delta_array = np.frombuffer(
#                     delta_data,
#                     dtype=np.dtype(delta_dict[name]["dtype"])
#                 ).reshape(tuple(delta_dict[name]["shape"]))
#
#                 # 应用增量
#                 new_array = orig_array + delta_array
#                 new_data = base64.b64encode(new_array.tobytes()).decode('utf-8')
#                 state_dict[name]["data"] = new_data
#
#         return json.dumps(state_dict)

class Group:
    """客户端组，管理串行训练 - pth文件版本"""

    def __init__(self, group_id, clients):
        self.id = group_id
        self.clients = clients
        self.leader = clients[0]  # 组长客户端
        self.temp_dir = f"/tmp/group_{group_id}_transfer"
        os.makedirs(self.temp_dir, exist_ok=True)

    def sequential_train(self, global_tsp_file, global_cep_file):
        """执行组内串行训练 - pth文件版本"""
        start_time = time.time()
        tsp_file = global_tsp_file
        cep_file = global_cep_file

        group_delta_tsp = None
        group_delta_cep = None

        # 客户端串行训练
        for client in self.clients:
            # 客户端训练
            delta_tsp, delta_cep, loss = client.train(tsp_file, cep_file)

            if delta_tsp is None or delta_cep is None:
                continue

            # 保存增量到文件
            delta_tsp_file = self._save_delta_to_file(delta_tsp, f"delta_tsp_{client.id}")
            delta_cep_file = self._save_delta_to_file(delta_cep, f"delta_cep_{client.id}")

            # 更新模型状态（通过应用增量）
            tsp_file = self._apply_delta_to_file(tsp_file, delta_tsp_file, f"updated_tsp_{client.id}")
            cep_file = self._apply_delta_to_file(cep_file, delta_cep_file, f"updated_cep_{client.id}")

            # 累积增量
            if group_delta_tsp is None:
                group_delta_tsp = delta_tsp
                group_delta_cep = delta_cep
            else:
                for name in delta_tsp:
                    if name in group_delta_tsp:
                        group_delta_tsp[name] += delta_tsp[name]
                    else:
                        group_delta_tsp[name] = delta_tsp[name]
                for name in delta_cep:
                    if name in group_delta_cep:
                        group_delta_cep[name] += delta_cep[name]
                    else:
                        group_delta_cep[name] = delta_cep[name]

        # 平均增量
        num_clients = len(self.clients)
        if group_delta_tsp:
            for name in group_delta_tsp:
                group_delta_tsp[name] = group_delta_tsp[name].float() / num_clients
                # group_delta_tsp[name] /= num_clients
        if group_delta_cep:
            for name in group_delta_cep:
                group_delta_cep[name] = group_delta_cep[name].float() / num_clients
                # group_delta_cep[name] /= num_clients

        # 保存平均增量到文件
        avg_delta_tsp_file = self._save_delta_to_file(group_delta_tsp, "avg_delta_tsp")
        avg_delta_cep_file = self._save_delta_to_file(group_delta_cep, "avg_delta_cep")

        # 清理临时文件
        # self._cleanup_temp_files()

        return avg_delta_tsp_file, avg_delta_cep_file, time.time() - start_time

    def _save_delta_to_file(self, delta_dict, prefix):
        """保存增量字典到文件"""
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.pth"
        filepath = os.path.join(self.temp_dir, filename)
        torch.save(delta_dict, filepath)
        return filepath

    def _apply_delta_to_file(self, state_file, delta_file, prefix):
        """应用增量到状态文件 - 修复版本"""
        try:
            # 检查状态文件是否存在
            if not os.path.exists(state_file):
                print(f"Error: State file not found: {state_file}")
                # 尝试在服务器目录中查找最新的全局模型文件
                server_dir = "/tmp/fed_server_transfer"
                if os.path.exists(server_dir):
                    tsp_files = glob.glob(os.path.join(server_dir, "global_tsp_*.pth"))
                    if tsp_files:
                        # 使用最新的全局模型文件
                        state_file = max(tsp_files, key=os.path.getmtime)
                        print(f"Using latest global model: {state_file}")
                    else:
                        raise FileNotFoundError(f"No global model files found in {server_dir}")
                else:
                    raise FileNotFoundError(f"Server directory not found: {server_dir}")

            # 检查增量文件是否存在
            if not os.path.exists(delta_file):
                raise FileNotFoundError(f"Delta file not found: {delta_file}")

            # 加载状态和增量
            state_dict = torch.load(state_file)
            delta_dict = torch.load(delta_file)

            print(f"Applying delta: state keys: {list(state_dict.keys())}, delta keys: {list(delta_dict.keys())}")

            # 应用增量
            for name in delta_dict:
                if name in state_dict:
                    state_dict[name] += delta_dict[name]
                    print(f"Applied delta to parameter: {name}, shape: {state_dict[name].shape}")
                else:
                    print(f"Warning: Parameter {name} not found in state dict")

            # 保存更新后的状态
            timestamp = int(time.time() * 1000)
            filename = f"{prefix}_{timestamp}.pth"
            filepath = os.path.join(self.temp_dir, filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            torch.save(state_dict, filepath)
            print(f"Saved updated state to: {filepath}")

            return filepath

        except Exception as e:
            print(f"Error applying delta: {e}")
            # 返回原始状态文件作为fallback
            return state_file

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up temp files for group {self.id}: {e}")

    def cleanup(self):
        """清理组资源"""
        self._cleanup_temp_files()
        try:
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except:
            pass

    def __del__(self):
        self.cleanup()


# def main():
#     """主函数"""
#     # 设置实验环境
#     setup_experiment()
#
#     # 先测试enclave启动
#     print("Testing enclave startup...")
#     test_enclave = SGXEnclaveManager(enclave_path='enclave_main.py')
#     # test_enclave = SGXEnclaveManager(enclave_path='/home/wbq/ETS_FL/pytorch/enclave_main.py')
#     if test_enclave.start_enclave():
#         print("Enclave test successful")
#         test_enclave.stop_enclave()
#     else:
#         print("Enclave test failed - check enclave_main.py")
#         return
#
#     # 初始化模型管理器
#     model_manager = ModelManager()
#
#     # 根据数据集确定类别数
#     num_classes = config.get_num_classes()
#
#     # 创建全局模型
#     global_tsp, global_cep, model_info = model_manager.create_models(
#         config.model_name, num_classes=num_classes
#     )
#
#     # 打印模型信息
#     print(f"\n模型信息:")
#     print(f"  名称: {config.model_name}")
#     print(f"  描述: {model_info['description']}")
#     print(f"  TSP参数量: {sum(p.numel() for p in global_tsp.parameters()):,}")
#     print(f"  CEP参数量: {sum(p.numel() for p in global_cep.parameters()):,}")
#     print(f"  总参数量: {sum(p.numel() for p in list(global_tsp.parameters()) + list(global_cep.parameters())):,}")
#
#     # 准备数据集
#     print(f"\n加载 {config.dataset_name} 数据集...")
#     fed_loader = FederatedDataLoader(
#         dataset_name=config.dataset_name,
#         num_clients=config.num_clients,
#         alpha=config.alpha,
#         data_root=config.data_dir,
#         train_batch_size=config.batch_size
#     )
#
#     # 获取测试集
#     test_loader = fed_loader.get_test_loader()
#
#     # 创建客户端和SGX enclaves
#     print("创建客户端和SGX enclaves...")
#     clients = []
#     feature_protector = FeatureProtector(
#         method=config.feature_protection_method,
#         sigma=config.feature_noise_sigma
#     )
#
#     for client_id in range(config.num_clients):
#         # 为每个客户端创建独立的SGX enclave
#         enclave_manager = SGXEnclaveManager(
#             memory_size=config.tee_memory,
#             enclave_path = './enclave_main.py'
#             # enclave_path='./pytorchexample.py'
#         )
#         enclave_manager.start_enclave()
#
#         # 创建客户端特定的TSP和CEP模型实例
#         client_model_tsp, client_model_cep, _ = model_manager.create_models(
#             config.model_name, num_classes=num_classes
#         )
#
#         client = SGXClient(
#             client_id,
#             client_model_tsp,
#             client_model_cep,
#             enclave_manager,
#             feature_protector,
#             local_epochs=config.local_epochs
#         )
#
#         # 设置数据加载器
#         train_loader, val_loader = fed_loader.get_client_dataloader(client_id)
#         client.set_dataloaders(train_loader, val_loader)
#
#         # 初始化客户端
#         if client.initialize():
#             print(f"客户端 {client_id} 初始化成功")
#             clients.append(client)
#         else:
#             print(f"客户端 {client_id} 初始化失败")
#
#     # 创建服务器和组
#     print("设置服务器和组...")
#     server = FederatedServer(global_tsp, global_cep)
#     server.initialize_global_model()
#     server.create_groups(clients, group_size=config.num_clients // config.num_groups)
#
#     # 运行联邦学习
#     print(f"\n开始联邦学习 ({config.rounds} 轮)...")
#     results = server.run_federated_learning(
#         config.rounds, clients, test_loader, config
#     )
#
#     # 保存结果
#     print("保存结果...")
#     result_dir = os.path.join(config.results_dir, f"{config.experiment_id}_{config.model_name}_{config.dataset_name}")
#     os.makedirs(result_dir, exist_ok=True)
#
#     # 保存结果数据
#     result_file = os.path.join(result_dir, "results.json")
#     save_results(results, result_file)
#
#     # 保存模型
#     torch.save(server.global_tsp_model.state_dict(), os.path.join(result_dir, "global_tsp.pth"))
#     torch.save(server.global_cep_model.state_dict(), os.path.join(result_dir, "global_cep.pth"))
#
#     # 可视化结果
#     print("生成可视化结果...")
#     plot_files = visualize_results(result_file, save_dir=result_dir)
#
#     print(f"\n实验完成! 结果保存在: {result_dir}")
#
#     # 清理
#     for client in clients:
#         client.enclave_manager.stop_enclave()

def main():
    """主函数"""
    setup_experiment()

    # 初始化模型管理器
    model_manager = ModelManager()
    num_classes = config.get_num_classes()

    # 创建全局模型
    global_tsp, global_cep, model_info = model_manager.create_models(
        config.model_name, num_classes=num_classes
    )

    # 准备数据集
    print(f"\n加载 {config.dataset_name} 数据集...")
    fed_loader = FederatedDataLoader(
        dataset_name=config.dataset_name,
        num_clients=config.num_clients,
        alpha=config.alpha,
        data_root=config.data_dir,
        train_batch_size=config.batch_size
    )

    # 获取测试集
    test_loader = fed_loader.get_test_loader()

    # 创建客户端和SGX enclaves
    print("创建客户端和SGX enclaves...")
    clients = []
    feature_protector = FeatureProtector(
        method=config.feature_protection_method,
        sigma=config.feature_noise_sigma
    )

    # 创建指定数量的客户端
    successful_clients = 0
    for client_id in range(config.num_clients):
        print(f"Creating client {client_id}...")

        # 为每个客户端创建独立的SGX enclave
        enclave_manager = SGXEnclaveManager(enclave_path='enclave_main.py')

        if not enclave_manager.start_enclave():
            print(f"Failed to start enclave for client {client_id}")
            continue

        # 创建客户端特定的TSP和CEP模型实例
        client_model_tsp, client_model_cep, _ = model_manager.create_models(
            config.model_name, num_classes=num_classes
        )

        client = SGXClient(
            client_id,
            client_model_tsp,
            client_model_cep,
            enclave_manager,
            feature_protector,
            local_epochs=config.local_epochs
        )

        # 设置数据加载器
        train_loader, val_loader = fed_loader.get_client_dataloader(client_id)
        client.set_dataloaders(train_loader, val_loader)

        # 初始化客户端
        if client.initialize():
            print(f"客户端 {client_id} 初始化成功")
            clients.append(client)
            successful_clients += 1
        else:
            print(f"客户端 {client_id} 初始化失败")
            enclave_manager.stop_enclave()

    if successful_clients == 0:
        print("No clients initialized successfully. Exiting.")
        return

    print(f"Successfully initialized {successful_clients} clients")

    # 运行联邦学习
    print("Starting federated learning...")

    # 创建服务器
    server = FederatedServer(global_tsp, global_cep)

    # 运行联邦学习过程 - 现在参数正确了
    acc_history, comm_cost, time_cost = server.run_federated_learning(
        config.rounds, clients, test_loader
    )

    # 保存结果
    result_file = save_results(acc_history, comm_cost, time_cost, config.__dict__)

    # 可视化结果
    visualize_results(result_file)

    # 清理资源
    for client in clients:
        client.cleanup()
        client.enclave_manager.stop_enclave()

    print("Federated learning completed")

if __name__ == "__main__":

    from config import config

    # 实验1：简单模型，CIFAR-10
    config.update(
        # 模型配置
        model_name = "resnet",  # "simple" 或 "resnet"
        dataset_name = "CIFAR10",  # "CIFAR10" 或 "CIFAR100"
        # 联邦学习配置
        num_clients = 3,
        num_groups = 2,  # 客户端组数量
        alpha = 0.3,  # Non-IID程度
        rounds = 3,  # 联邦学习轮次
        # 训练配置
        batch_size = 32,
        local_epochs = 1,  # 客户端本地训练轮次
        # TEE配置
        tee_memory = 128,  # MB
        feature_protection_method = "gaussian",  # 特征保护方法
        feature_noise_sigma = 0.01  # 高斯噪声标准差
    )

    # 实验2：ResNet模型，CIFAR-100
    # config.update(
    #     model_name="resnet",
    #     dataset_name="CIFAR10",
    #     num_clients=20,
    #     rounds=50,
    #     batch_size=32,
    #     tee_memory=256  # ResNet需要更多内存
    # )

    # 运行主程序
    main()

