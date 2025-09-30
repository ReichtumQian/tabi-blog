+++

title = "PyTorch 网络结构基础"

date = "2025-09-22"

[taxonomies]

tags = ["PyTorch", "Python"]

+++



在 PyTorch 中，无论是整个网络、一个卷积层，还是一个包含多个层的复杂模块，它们都继承自同一个基类：`torch.nn.Module`​。`nn.Module`​ 可以包含其他的 `nn.Module`​，这使得网络架构可以像套娃一样层层嵌套，形成一个树状结构。

---

## PyTorch 网络结构基础

**网络层 Layers**：网络层是神经网络中最基本的计算单元，负责执行具体的数学运算。常见的网络层类型包括：

- ​`Conv2d`​: 二维卷积层，用于提取图像特征，是 CNN 的核心。例如 `Conv2d(3, 64, kernel_size=(3, 3), ...)`​ 表示输入通道为 3 (RGB图像)，输出通道为 64，卷积核大小为 3x3。
- ​`BatchNorm2d`​: 二维批量归一化层，用于加速模型训练，提高稳定性。
- ​`ReLU`​: 激活函数层，引入非线性，使网络能够学习更复杂的模式。
- ​`Linear`​: 全连接层，通常用于网络的末端，进行分类或回归。
- ​`Identity`​: 占位层，它什么也不做，原样输出输入。在 ResNet 中有时用于保持结构一致性。

**网络层名 Layer Names**：网络层名是你在代码中给每个网络层或容器赋予的变量名。在 PyTorch 中的模型摘要中，括号内的为网络层名。网络层名是在模型初始化时被赋予的

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 就是层名 (conv1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        # self.pool 就是层名 (pool)
        self.pool = nn.MaxPool2d(2)

# 我们可以用 print(model) 打印出一个模型的架构
```

**网络层容器 Containers**：网络层容器本身也是一种 `nn.Module`​，它的主要作用是像一个 "盒子" 或 "工具箱"，用来组织和管理其他的 `nn.Module`​（包括其他容器和计算层）。我们这里介绍两种容器：

- ​`Sequential`​: 顺序容器。这是最常见的容器，它会按照你添加模块的顺序，依次执行内部的模块。
- 自定义容器 ( 例如 `ResNet`​, `BasicBlock`​): 除了 `Sequential`​ 这种现成的容器，我们还可以通过自己定义一个类并继承 `nn.Module`​ 来创建更复杂的自定义容器。

**Sequential 容器中的层名**：默认情况下 `Sequential`​ 容器从 `0`​ 开始给每层容器起名。例如

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
    		nn.Conv2d(1, 20, 5),
    		nn.ReLU(),
    		nn.Conv2d(20, 64, 5),
    		nn.ReLU()
		)
		self.layer2 = nn.Sequential(OrderedDict([
    		('convolution_layer_1', nn.Conv2d(1, 20, 5)),
    		('activation_1', nn.ReLU()),
    		('convolution_layer_2', nn.Conv2d(20, 64, 5)),
    		('activation_2', nn.ReLU())
		]))
model = MyModel()
print(model)
```

输出应该如下：

```bash
MyModel(
  (layer1): Sequential(
    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
    (3): ReLU()
  )
  (layer2): Sequential(
    (convolution_layer_1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    (activation_1): ReLU()
    (convolution_layer_2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
    (activation_2): ReLU()
  )
)
```

---

## State Dict 方法及其用途

**State Dict 方法**：`model.state_dict()`​ 会返回一个 `OrderedDict`​，Key 为每层参数的名称，Value 是每层参数的值。例如前面的 `MyModel`​ 的 `state_dict()`​ 为：

```python
{
    # 来自 layer1
    'layer1.0.weight':  torch.Size([20, 1, 5, 5]),
    'layer1.0.bias':    torch.Size([20]),
    'layer1.2.weight':  torch.Size([64, 20, 5, 5]),
    'layer1.2.bias':    torch.Size([64]),

    # 来自 layer2（使用了 OrderedDict 显式命名）
    'layer2.convolution_layer_1.weight': torch.Size([20, 1, 5, 5]),
    'layer2.convolution_layer_1.bias':   torch.Size([20]),
    'layer2.convolution_layer_2.weight': torch.Size([64, 20, 5, 5]),
    'layer2.convolution_layer_2.bias':   torch.Size([64])
}
```

**保存与加载模型**：`model.state_dict()`​ 最基础的用途是保存和加载模型，使用 `torch.save()`​ 保存，使用 `model.load_state_dict()`​ 加载

```python
# 保存
torch.save(model.state_dict(), 'my_model.pth')

# 加载（必须先创建相同结构的模型）
model = MyModel()
model.load_state_dict(torch.load('my_model.pth'))
```

**给另一个模型赋值**：如果两个模型结构完全一致（类定义相同），则可以直接赋值

```python
model1 = MyModel()
model2 = MyModel()

# 将 model1 的参数复制给 model2
model2.load_state_dict(model1.state_dict())
```

**部分赋值给另一个模型（微调常用）** ：即使两个模型**不完全相同**，只要部分层的名字和形状匹配，也可以**选择性加载**：

```python
# 假设 model_pretrained 是一个大型预训练模型
# model_new 是你的新模型，但共享部分结构（比如前面的卷积层）

pretrained_dict = model_pretrained.state_dict()
new_model_dict = model_new.state_dict()

# 过滤出新模型中也存在的、且形状匹配的参数
filtered_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in new_model_dict and v.shape == new_model_dict[k].shape
}

# 更新新模型的 state_dict
new_model_dict.update(filtered_dict)
model_new.load_state_dict(new_model_dict)
```

---

## ResNet-18 网络结构解析

我们以 `torchvision`​ 中的 `resnet-18`​ 为例，看看其网络结构。

- 直接成员变量：`self.conv1`​、`self.bn1`​、`self.relu`​、`self.maxpool`​、`self.layer1/2/3/4`​、`self.avgpool`​、`self.fc`​
- 容器结构：4 个 `self.layer`​ 都使用了 `nn.Sequential`​，并且内部使用了自定义的 `BasicBlock`​，每个 `BasicBlock`​ 还各有不同。

```bash
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): Identity()

    (layer1): Sequential(
        (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )

    (layer2): Sequential(
        (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
                (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )

    (layer3): Sequential(
        (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
                (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )

    (layer4): Sequential(
        (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )

    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=512, out_features=10, bias=True)
)

```
