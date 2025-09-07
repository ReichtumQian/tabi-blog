+++

title = "Pytorch 求导操作总结"

date = "2025-09-07"

[taxonomies]

tags = ["Pytorch", "Python"]

+++

自动求导（Autograd）是 Pytorch 框架的基石，下面我们对其思想、方法进行总结。

---

## 计算图 Computational Graph

**计算图的概念**：当我们对一个 Tensor 执行任何操作时，Pytorch 后台会默默构建一个<u>有向无环图</u> Graph 来记录这些操作。图的<u>节点为 Tensors</u>；<u>边为函数 Operations</u>，它们接收输入的 Tensors 并计算输出的 Tensors。一个典型的计算图如下所示：

```text
x ---.
     v
     * --> a ---.
     ^          v
y ---'          + --> z
                ^
w --------------'
```

**自动求导**：Pytorch 的自动求导引擎 `autograd`​ 通过上面的计算图，从最终的输出 `z`​ 开始，使用链式法则（`z.backward()`​），反向追溯到每个输入参数 `x`​、`y`​、`w`​。从而计算 `z`​ 关于这些输入值的梯度。

**requires_grad 属性**：`requires_grad`​ 是 `torch.Tensor`​ 的一个 `bool`​ 类型属性，默认值为 `False`​。如果为 `True`​，则会追踪对这个 `Tensor`​ 的所有操作。任何需要学习的参数（如 `weight`​ 和 `bias`​），它们本质上也是 `torch.Tensor`​，必须把它们的 `requires_grad`​ 设置为 `True`​。

```python
x = torch.tensor(2.0, requires_grad=True) # 初始化一个需要 grad 的 Tensor
x.requires_grad_(True) # 设置已存在的 Tensor 需要 grad
```

**grad 属性**：对于 `requires_grad`​ 的 Tensor，最终计算的梯度会被保存在 `.grad`​ 属性中

> 例如上图中我们希望计算 `z`​ 关于 `x`​ 的导数，则必须把 `x.requires_grad`​ 设置为 `True`​。

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=False)  # 注意：False
w = torch.tensor(4.0, requires_grad=False)  # 注意：False

a = x * y   # a.requires_grad = True（因为 x 需要梯度）
z = a + w   # z.requires_grad = True

z.backward()

print(x.grad)  # tensor(3.) —— 正确：dz/dx = dz/da * da/dx = 1 * y = 3
print(y.grad)  # None —— 因为 requires_grad=False
print(w.grad)  # None —— 同上
```

---

## 梯度的计算：Backward 与 Autograd

**Backward 函数**：`.backward()`​ 函数会从当前的 `Tensor`​ 出发（通常是最终的 `loss`​），沿着计算图反向传播，计算所有 `requires_grad = True`​ 的叶子节点 Tensor 的梯度。对于中间节点，其会计算关于它们的梯度，但是用完后会被直接释放不被保存，除非显式调用 `.retain_grad()`​。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)  # leaf
y = torch.tensor(3.0, requires_grad=True)  # leaf

a = x * y   # intermediate node
z = a ** 2  # output

print("a.is_leaf:", a.is_leaf)   # False
print("z.is_leaf:", z.is_leaf)   # False

z.backward()

print("x.grad:", x.grad)  # tensor(36.)
print("y.grad:", y.grad)  # tensor(24.)
print("a.grad:", a.grad)  # None ← 默认不保存！
```

**梯度的累积**：Pytorch 中默认梯度会累积，即每次调用 `.backward()`​ 后，Pytorch 会把新计算的梯度加到原来的 `.grad`​ 上

```python
x = torch.tensor([2.0], requires_grad=True)

# 第一次计算和反向传播
y1 = x**2
y1.backward()
print(f"After first backward, x.grad: {x.grad}") # 输出: tensor([4.]) (dy1/dx = 2x = 4)

# 第二次计算和反向传播
y2 = x**3
y2.backward()
# 新的梯度 (dy2/dx = 3x^2 = 12) 会被加到旧的梯度上
print(f"After second backward, x.grad: {x.grad}") # 输出: tensor([16.]) (4 + 12)
```

**torch.autograd.grad()** ：`autograd.grad()`​ 是比 `backward()`​ 更加底层的函数，其直接调用 `autograd`​ 引擎返回所需的梯度，而不是像 `.backward()`​ 一样将梯度填充到 `.grad`​ 属性中。

```python
# torch.autograd.grad(outputs, inputs, create_graph=False, ...)
# outputs：相当于 .backward() 的调用者
# inputs：要相对于哪个 Tensor 求导
# create_graph：如果设置为 True，则返回的梯度本身也会构成计算图的一部分，从而计算高阶导数

import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2 + y**3 # z = 4 + 27 = 31

# 使用 torch.autograd.grad 计算 z 相对于 x 和 y 的梯度
# dz/dx = 2x = 4, dz/dy = 3y^2 = 27
gradients = torch.autograd.grad(z, (x, y)) # (4, 27)

print(f"Returned gradients (dz/dx, dz/dy): {gradients}")
print(f"x.grad is still: {x.grad}") # 输出 None，因为它没有被修改
print(f"y.grad is still: {y.grad}") # 输出 None


```

**高阶求导示例**：`autograd.grad()`​ 还可以将求导作为一个操作放入计算图中。例如用 `z`​ 关于 `x`​ 求导得到 `grad_x`​，这会创建一个新的计算图，包含了 `x`​ 到 `grad_x`​ 的计算关系。

```python
# 我们想求 d^2z / dx^2
# 首先，计算一阶导数 dz/dx，并让这个过程可导
grad_x, = torch.autograd.grad(z, x, create_graph=True) 

print(f"\nFirst derivative dz/dx: {grad_x}") # grad_x = 2x

# 现在对一阶导数 grad_x 再次求导
# d(grad_x)/dx = d(2x)/dx = 2
second_grad_x, = torch.autograd.grad(grad_x, x)

print(f"Second derivative d^2z/dx^2: {second_grad_x}")

# --- 第一层：原始计算 ---
# x ----> [ Pow(2) ] ----> z
#           ^
#           |
#           '--- (第一次反向传播, create_graph=True)

#  --- 第二层：导数的计算 ---
# x ----> [ Mul(2) ] ----> grad_x
#           ^
#           |
#           '--- (第二次反向传播) ---> second_grad_x (值为 2)
```

## 清空梯度

由于 Pytorch 的默认行为是累积梯度，因此在神经网络每个 `batch`​ 开始前，我们必须手动清空上一轮积累的梯度，常见的方法有两种：

**手动清零**：对 Tensor 的 `grad`​ 属性调用 `.zero_()`​方法，这会遍历模型的所有参数并逐个清零

```python
x.grad.zero_() # 使用 in-place 的 zero_() 方法清空梯度
```

**使用优化器清零**：这是标准做法，Pytorch 的优化器会接收模型的所有参数，调用 `optimizer.zero_grad()`​ 即可自动清零模型的参数

```python
# 假设有一个模型和优化器
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 在训练循环的开始
optimizer.zero_grad()
```

---

‍
