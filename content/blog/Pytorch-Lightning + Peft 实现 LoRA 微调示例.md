+++

title = "Pytorch-Lightning + Peft 实现 LoRA 微调示例"

date = "2025-09-15"

[taxonomies]

tags = ["LoRA", "Fine Tuning"]

+++

‍

---

## LoRA 思想回顾

LoRA 假设在微调过程中，权重的变化是一个 low-rank 的矩阵，因此其冻结模型原始权重 $W_0$，并在旁边注入两个小的、可训练的 low-rank 矩阵 $A$ 和 $B$ 来模拟权重的更新 $\Delta W$：

$$
W_{tuned}=W_0+\Delta W=W_0+B A
$$

其中 $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$，其中 $r$ 为指定的权重矩阵的秩。

---

## Pytorch-Lightning + Peft 实现 LoRA

`peft`​ （Parameter-Efficient Fine-Tuning）是 Hugging Face 公司实现的用于高效微调的库。其核心工具是 `get_peft_model`​ 函数（95% 情况下够用）。下面以一个最简单的例子说明如何使用 `peft`​ 库实现 LoRA。

**Step 1. 准备一个 Pytorch 模型**：`peft`​ 的 `get_peft_model`​ 函数是作用于一个现有的模型，因此我们需要先实现一个最简化的模型

```python
import torch
import torch.nn as nn

# 1. 创建一个极其简单的 PyTorch 模型
# 它包含两个线性层，就是一个普通的神经网络
original_model = nn.Sequential(
    nn.Linear(in_features=128, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=10) # 假设最后输出10个分类
)

# 看看模型有多少参数
total_params = sum(p.numel() for p in original_model.parameters())
print(f"原始模型总参数量: {total_params:,}")

# 原始模型总参数量: 35,594
```

**Step 2. 定义 LoRA 配置并应用**：我们需要定义一个 `LoraConfig`​ 对象，并且包含 `LoRA`​ 的配置信息。其中 `target_modules`​ 是一个字符串的列表，该 `LoRA`​ 配置会作用于任何 `name`​ 中**包含**这些字符串的层。

```python
from peft import get_peft_model, LoraConfig, TaskType

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # 任务类型，这里是序列（或特征）分类
    r=4,                        # LoRA的秩，r越小，新增的参数越少。通常设为4, 8, 16
    lora_alpha=32,              # LoRA的缩放因子
    lora_dropout=0.1,           # Dropout 比例
    target_modules=["Linear"]   # 指定要应用 LoRA 的模块类型。这里我们对所有 nn.Linear 层都应用
)


# 3. 使用 get_peft_model 将 LoRA 应用到原始模型上
peft_model = get_peft_model(original_model, lora_config)

# 关键时刻：打印可训练参数！
peft_model.print_trainable_parameters()

# trainable params: 3,120 || all params: 38,714 || trainable%: 8.0591
```

**Step 3. 像平常一样训练模型**：`peft`​ 库最强大的地方是其只对模型进行了修改，训练流程完全不变

```python
# 4. 创建一个简单的优化器和虚拟数据
optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 创建一些假数据来模拟训练
dummy_input = torch.randn(16, 128)  # 16个样本，每个样本128维
dummy_labels = torch.randint(0, 10, (16,)) # 16个标签，0-9之间

print("\n--- 开始一个简单的训练步骤 ---")

# 把模型设置为训练模式
peft_model.train()

# 前向传播
outputs = peft_model(dummy_input)

# 计算损失
loss = loss_fn(outputs, dummy_labels)
print(f"计算出的损失: {loss.item()}")

# 反向传播和优化
# PyTorch 会自动只计算和更新可训练参数的梯度
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("--- 训练步骤完成！只有LoRA参数被更新了 ---")
```
