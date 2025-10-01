+++

title = "代码阅读：KL 散度正则项计算"

date = "2025-10-01"

[taxonomies]

tags = ["PyTorch"]

+++

‍

我们考虑下面的 KL 散度正则 Loss，在微调过程中加入新数据上的 KL 散度以缓解遗忘

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fine-tune}}
  + \mathbb{E}_{x \sim \mathcal{D}_{\text{fine-tune}}} \left[ D_{KL}(\pi_{\theta}(\cdot|x) || \pi_{\theta_0}(\cdot|x)) \right],
$$

```python
def train_epoch_kl_reg(model, model_old, x_train, y_train, optimizer, criterion, kl_lambda):
    # 设置模型行为模式
	model.train()
    model_old.eval()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss_new = criterion(outputs, y_train)
    with torch.no_grad():
        logits_old_ref = model_old(x_train)
        probs_old_ref = F.softmax(logits_old_ref, dim=1)
    log_probs_current = F.log_softmax(outputs, dim=1)
    kl_penalty = F.kl_div(log_probs_current, probs_old_ref, reduction='batchmean', log_target=False)
    total_loss = loss_new + kl_lambda * kl_penalty
    total_loss.backward()
    optimizer.step()
```

‍

**设置模型行为模式**：`model.train()`​ 和 `model.eval()`​ 用于控制模型的行为模式，模型默认处于 `train`​ 模式。具体而言，它们将所有子模块的 `self.training`​ 设置为 `True`​ 或者 `False`​，其在以下层行为不同：

- `Dropout`​ 层：Dropout 是一种正则化技术，用于防止模型过拟合。在训练期间，它会随机地将输入张量中的一部分元素置为零（即“丢弃”神经元）
- `BatchNorm`​ 层：批量归一化（Batch Normalization）层可以加速模型收敛并提高稳定性，其计算当前批次 (mini-batch) 的均值和方差，使用这个批次的均值和方差来<u>归一化当前批次的输入数据</u>。同时，它还会<u>更新一个全局的“运行统计量”</u>（`running_mean`​ 和 `running_var`​）。这个运行统计量在评估阶段会派上用场。

|功能 / 模块|model.train() (训练模式)|model.eval() (评估模式)|
| --------------| -------------------------------------------------------------------------------------| ------------------------------------------------------------------------|
|主要用途|用于模型的训练阶段|用于模型的验证、测试或推理阶段|
|Dropout 层|激活。在前向传播中，会以指定概率 p 随机“丢弃”神经元。|关闭。在前向传播中，所有神经元都会被使用。|
|BatchNorm 层|1. 使用当前批次数据的均值和方差进行归一化。<br />2. 会更新层的 `running_mean`​ 和 `running_var`​，用于后续评估。|1. 使用训练阶段学习到的 `running_mean`​ 和 `running_var`​ 进行归一化。<br />2. 不会更新这些统计值。|
|梯度计算|默认开启（但由 torch.no\_grad() 控制）|默认开启（但由 torch.no\_grad() 控制）|

> `model.eval()`​ 和 `torch.no_grad()`​ 关系：它们容易被混淆，但实际功能是正交的，经常配合使用：
>
> - `model.eval()`​：只改变 `Dropout`​ 和 `BatchNorm`​ 等层的行为模式，不影响梯度计算
> - `torch.no_grad()`​：其关闭该代码块中所有 PyTorch 操作的梯度计算
>
> 注意：即使使用了 `model.eval()`​，在 `loss.backward()`​ 时还是会更新模型参数的 `.grad`​ 属性，因此一定要用 `no_grad()`​！

**torch.nn.functional 函数**：`torch.nn.functional`​ 提供了各类函数，这段代码中用到了以下几个：

- `F.softmax`​：将 logits 映射为概率分布

```python
import torch
import torch.nn.functional as F

# 假设这是一个模型对3个样本、4个类别的原始输出 (logits)
# shape: [batch_size, num_classes]
logits = torch.tensor([
    [1.0, 3.0, 0.5, 2.0],  # 样本1
    [0.1, -1.0, 2.0, 0.5], # 样本2
    [-2.0, -1.0, -0.5, -3.0] # 样本3
])

# 对每一个样本，在“类别”维度上应用 softmax
# dim=1 表示对每一行进行操作
probabilities = F.softmax(logits, dim=1)

print("Logits:")
print(logits)
print("\nProbabilities (after F.softmax):")
print(probabilities)
print("\nSum of probabilities for each sample:")
print(probabilities.sum(dim=1))

# Logits:
# tensor([[ 1.0000,  3.0000,  0.5000,  2.0000],
#         [ 0.1000, -1.0000,  2.0000,  0.5000],
#         [-2.0000, -1.0000, -0.5000, -3.0000]])

# Probabilities (after F.softmax):
# tensor([[0.0863, 0.6375, 0.0521, 0.2241],
#         [0.1174, 0.0390, 0.7858, 0.1778],
#         [0.0531, 0.1444, 0.2384, 0.0195]])

# Sum of probabilities for each sample:
# tensor([1.0000, 1.0000, 1.0000])
```

- `F.log_softmax()`​：先计算 `F.softmax`​ 再取自然对数

```python
# 使用上面的 logits
log_probabilities = F.log_softmax(logits, dim=1)

print("\nLog Probabilities (after F.log_softmax):")
print(log_probabilities)

# Log Probabilities (after F.log_softmax):
# tensor([[-2.4499, -0.4499, -2.9499, -1.4999],
#         [-2.1423, -3.2423, -0.2423, -1.7223],
#         [-2.9351, -1.9351, -1.4351, -3.9351]])
```

- `F.kl_div`​：计算 KL 散度

$$
D_{KL}(P||Q)=\sum_iP(i)\log\left(\frac{P(i)}{Q(i)}\right)=\sum_iP(i)(\log P(i)-\log Q(i))
$$

```python
# F.kl_div(input, target, reduction='batchmean', log_target=False)
# input 必须是对数概率！即 F.log_softmax 的输出
# target 必须是标准概率，即 F.softmax 的输出
# reduction：'batchmean' 表示计算出的损失总和会除以 batch size（张量的第一个维度）
# 	         'mean' 表示损失会按元素数量取平均（张量的 numel）
#            'sum' 表示对损失求和
# log_target：如果 target 也是对数概率，则需要将该参数设置为 True
```
