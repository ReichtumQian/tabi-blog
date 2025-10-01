+++

title = "Peft 库：LoRA 对网络模型结构的修改"

date = "2025-10-01"

[taxonomies]

tags = ["PyTorch", "Peft Library"]

+++

‍

---

## Peft 库对网络结构的修改

PEFT 对模型结构的影响不是“修改”，而是“包裹”和“注入”。它在不改变原始模块内部代码的情况下，动态地在其外部添加新的功能层。我们以最主流的 PEFT 方法 LoRA 为例。

**Linear 层的例子**：我们构造一个线性层，可以将其打印出来

```python
import torch.nn as nn

original_model = nn.Linear(in_features=100, out_features=200)
print(original_model)

# Linear(in_features=100, out_features=200, bias=True)
```

接着我们对其应用 `LoRA`​ ：

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(r=8, target_modules=["linear"])
peft_model = get_peft_model(original_model, lora_config)
print(peft_model)
```

运行程序会输出类似下面的结果，其中 `base_layer`​ 是原来的模型，其余的为 `LoRA`​ 额外添加的层

```bash
Linear(
  (base_layer): Linear(in_features=100, out_features=200, bias=True)
  (lora_dropout): ModuleDict(...)
  (lora_A): ModuleDict(
    (default): Linear(in_features=100, out_features=8, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=8, out_features=200, bias=False)
  )
  ...
)
```

---

## Peft 库对 require_grad 的修改

Peft 库最核心的工作是调整了所有参数的 `require_grad`​ 属性，例如我们创建一个简单的模型：

```python
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

# --- 准备工作 ---

# 1. 创建一个简单的原始模型
original_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    r=4,
    target_modules=["0", "2"] # 明确指定要改造的线性层
)

# 3. 获取 PEFT 模型
peft_model = get_peft_model(original_model, lora_config)


# --- 开始检查 requires_grad ---

print("="*60)
print("1. 检查【原始模型】的参数状态 (改造前):")
print("="*60)
# 默认情况下，模型的所有参数都应该是可训练的
for name, param in original_model.named_parameters():
    print(f"{name:<45} | requires_grad: {param.requires_grad}")

print("\n" + "="*60)
print("2. 检查【PEFT 模型】的参数状态 (改造后):")
print("="*60)
# PEFT 会冻结大部分参数，只保留 LoRA 部分可训练
for name, param in peft_model.named_parameters():
    # 为了更清晰地展示，我们只筛选出包含 lora 或 base_layer 的参数名
    if 'lora' in name or 'base_layer' in name:
        print(f"{name:<45} | requires_grad: {param.requires_grad}")

print("\n" + "="*60)
print("PEFT 模型的可训练参数摘要:")
peft_model.print_trainable_parameters()
print("="*60)
```

输出如下：

```bash
============================================================
1. 检查【原始模型】的参数状态 (改造前):
============================================================
0.weight                                      | requires_grad: True
0.bias                                        | requires_grad: True
2.weight                                      | requires_grad: True
2.bias                                        | requires_grad: True

============================================================
2. 检查【PEFT 模型】的参数状态 (改造后):
============================================================
base_model.model.0.base_layer.weight          | requires_grad: False
base_model.model.0.base_layer.bias            | requires_grad: False
base_model.model.0.lora_A.default.weight      | requires_grad: True
base_model.model.0.lora_B.default.weight      | requires_grad: True
base_model.model.2.base_layer.weight          | requires_grad: False
base_model.model.2.base_layer.bias            | requires_grad: False
base_model.model.2.lora_A.default.weight      | requires_grad: True
base_model.model.2.lora_B.default.weight      | requires_grad: True

============================================================
PEFT 模型的可训练参数摘要:
trainable params: 340 || all params: 345 || trainable%: 98.5507
============================================================
```
