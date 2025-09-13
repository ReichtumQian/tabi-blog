+++

title = "Pytorch 中的 Dataset、DataLoader"

date = "2025-09-13"

[taxonomies]

tags = ["PyTorch", "Python", "Machine Learning"]

+++

`Pytorch`​ 中的数据管理由 `Dataset`​ 和 `DataLoader`​ 两个类实现。

---

## Pytorch 中的数据处理

**Pytorch 数据处理的步骤**：`Pytorch`​ 中处理和加载数据一般需要经历以下步骤

- 加载原始数据：从硬盘读取数据（例如图片、文本、CSV 文件）
- 预处理数据：将原始数据转换成 Tensor，进行必要的变换（如尺寸调整、归一化）
- 分批（Batching）：将整个数据集分成一个个小批次（mini-batch）
- 迭代：将这些批次数据送入模型进行训练或评估

**Dataset 与 DataLoader 的分工**：`Pytorch`​ 中主要用 `Dataset`​ 和 `DataLoader`​ 两个类实现了上述操作

- `Dataset`​：负责<u>加载和预处理单个数据样本</u>，使用 `__getitem__`​ 读取每一个数据点。
- `DataLoader`​：负责从 `Dataset`​ 中取出数据，并将其打包成批次以供训练

---

## torch.utils.data.Dataset 类

**Dataset 类基本使用**：`Dataset`​ 是一个抽象类，我们需要创建自己的类来继承它，并实现两个方法：

- `__len__(self)`​：返回数据集样本的总数。`DataLoader`​ 会用它来确定迭代的次数。
- `_getitem__(self, idx)`​：接收一个索引 `idx`​，返回数据集中对应的<u>一个样本</u>。数据加载和预处理（`transform`​）一般在这里完成。

**示例：从文件夹读取图片**：假设我们的图片存储在如下目录中：

```bash
/data/my_images/
├── cat.0.jpg
├── cat.1.jpg
...
├── dog.0.jpg
├── dog.1.jpg
...
```

我们可以创建一个 `CustomImageDataset`​

```python
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_labels = [] # 用一个列表存储 (图片路径, 标签)
        for file_name in os.listdir(data_dir):
            if file_name.startswith('cat'):
                label = 0
            elif file_name.startswith('dog'):
                label = 1
            else:
                continue
            self.img_labels.append((os.path.join(data_dir, file_name), label))
        
        # 保存 transform
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 1. 获取图片路径和标签
        img_path, label = self.img_labels[idx]
        # 2. 读取图片
        image = Image.open(img_path).convert('RGB')
        # 3. 如果定义了 transform，则对图片进行处理
        if self.transform:
            image = self.transform(image)
        # 4. 返回处理后的图片和对应的标签
        return image, torch.tensor(label, dtype=torch.long)
```

**数据变换与增强 transform**：原始数据很少能直接送入模型。图片需要被统一尺寸、转换为 Tensor、并进行归一化；文本需要被分词、转换为 ID。`transform`​ 就是用来完成这些预处理工作的。例如 `torchvision.transforms`​ 模块为图像处理提供了大量现成的工具

```python
from torchvision import transforms

# 定义一系列变换
# 1. 将图片缩放到 224x224
# 2. 随机水平翻转（数据增强）
# 3. 转换为 Tensor（将 HWC 的 PIL Image 或 NumPy 数组转换为 CHW 的 Tensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]）
# 4. 归一化
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化 Dataset 时传入 transform
my_dataset = CustomImageDataset(data_dir='/data/my_images/', transform=image_transform)

# 现在，当我们通过 my_dataset[0] 获取数据时，得到的图片已经是经过上述所有变换后的 Tensor 了。
```

---

## torch.utils.data.DataLoader 类

**DataLoader 的基本使用**：`DataLoader`​ 是一个迭代器，它从 `Dataset`​ 中自动抓取数据，打包成批次。其包含如下核心参数：

- `dataset`: 我们刚刚创建的 `Dataset`​ 实例。
- `batch_size`​ (int): 每个批次包含的样本数。默认为 1。
- `shuffle`​ (bool): 是否在每个 epoch 开始时打乱数据顺序。训练时通常设为 `True`​，验证/测试时设为 `False`​。
- `num_workers`​ (int): 用于数据加载的子进程数。`0`​ 表示在主进程中加载。大于 `0`​ 的值可以显著加速数据加载，避免 GPU 等待。
- `collate_fn`​ (callable, optional): 用于将多个样本合并成一个批次的函数。下面会重点讲解。
- `pin_memory`​ (bool): 如果为 `True`​，数据加载器会将张量复制到 CUDA 固定内存中，这可以加快数据到 GPU 的传输速度。

**示例：创建和使用 DataLoader**

```python
from torch.utils.data import DataLoader

# 假设 my_dataset 已经创建好了
# 创建 DataLoader
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=32,
    shuffle=True,      # 训练数据需要打乱
    num_workers=4,     # 使用4个子进程加载数据
    pin_memory=True
)

# 如何使用 DataLoader 进行迭代
num_epochs = 10
for epoch in range(num_epochs):
    # train_loader 是一个可迭代对象
    for batch_images, batch_labels in train_loader:
        # 在这里，batch_images 和 batch_labels 就是一个批次的数据
        # batch_images 的形状通常是 (batch_size, channels, height, width) -> (32, 3, 224, 224)
        # batch_labels 的形状通常是 (batch_size) -> (32)
        
        # 将数据移动到 GPU (如果可用)
        # batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # ... 接下来是模型前向传播、计算损失、反向传播等步骤 ...
        
        # 打印一个批次的形状看看
        print(f"Epoch {epoch+1}, Batch Image Shape: {batch_images.shape}, Batch Label Shape: {batch_labels.shape}")
        break # 这里只演示一个批次
    break
```

**collate_fn 自定义批次合并逻辑**：`DataLoader`​ 的默认合并函数 `default_collate`​ 简单地使用 `torch.stack()`​ 将样本列表堆叠成一个批次，其使用场景是 `Dataset`​ 返回的每个样本都有相同的形状。但是，<u>如果样本形状不一，</u>​<u>​`default_collate`​</u>​<u> 就会报错</u>。最典型的例子是自然语言处理（NLP），每个句子的长度（词元数量）都不同。此时就需要自定义 `collate_fn`​，其工作逻辑为：

- `DataLoader`​ 从 `Dataset`​ 中取出 `batch_size`​ 个样本，形成一个列表，例如 `[sample1, sample2, ..., sample_batch_size]`​。
- `DataLoader`​ 将这个列表传递给 `collate_fn`​ 函数。
- `collate_fn`​ 函数负责处理这个列表，并返回一个或多个已经正确打包好的批次 Tensor。

**示例：处理不同长度的序列**：假设我们有一个 `Dataset`​ 返回的是 `(序列, 标签)`​，其中序列是不同长度的 Tensor。

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. 创建一个返回可变长度序列的数据集
class VariableLenDataset(Dataset):
    def __init__(self):
        self.data = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5]), 1),
            (torch.tensor([6, 7, 8, 9]), 0),
            (torch.tensor([10]), 1)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 2. 定义自定义的 collate_fn
def custom_collate_fn(batch):
    """
    Args:
        batch: 一个列表，其中每个元素都是 Dataset 的一个返回值，即 (sequence, label)
               例如: [(tensor([1,2,3]), 0), (tensor([4,5]), 1)]
    """
    # 将序列和标签分开
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 对序列进行填充(padding)，使它们长度一致
    # pad_sequence 会自动找到批次中最长的序列，并将其他序列填充到该长度
    # batch_first=True 表示返回的 Tensor 形状为 (batch_size, max_seq_length)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # 将标签列表转换为 Tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_sequences, labels

# 3. 在 DataLoader 中使用 custom_collate_fn
my_dataset = VariableLenDataset()
my_loader = DataLoader(
    dataset=my_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=custom_collate_fn # 关键在这里！
)

# 4. 迭代并查看结果
for seq_batch, label_batch in my_loader:
    print("Padded Sequences Batch:\n", seq_batch)
    print("Shape:", seq_batch.shape)
    print("Labels Batch:\n", label_batch)
    print("Shape:", label_batch.shape)
    print("-" * 20)

# 预期输出：
# Padded Sequences Batch:
#  tensor([[1, 2, 3],
#          [4, 5, 0]])  <-- 第二个序列被填充了一个0
# Shape: torch.Size([2, 3])
# Labels Batch:
#  tensor([0, 1])
# Shape: torch.Size([2])
# --------------------
# Padded Sequences Batch:
#  tensor([[6, 7, 8, 9],
#          [10, 0, 0, 0]]) <-- 第二个序列被填充了三个0
# Shape: torch.Size([2, 4])
# Labels Batch:
#  tensor([0, 1])
# Shape: torch.Size([2])
# --------------------
```
