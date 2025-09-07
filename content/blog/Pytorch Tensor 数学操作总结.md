+++

title = "Pytorch Tensor 数学操作总结"

date = "2025-09-05"

[taxonomies]

tags = ["PyTorch", "Python"]

+++

---

## 逐元素运算（Element-Wise Operations）

逐元素运算指的是对两个形状相同的 Tensor 的对应元素进行操作。

```python
# 假设 x 和 y 是同 size 的 tensor，a 为 float
z = x + y
z = x - y
z = x * y  # 逐元素乘法
z = x / y  # 逐元素除法
z = x ** a # 逐元素取 a 次方
z = torch.sqrt(x) # 逐元素开根号
```

---

## 广播运算（Broadcasting）

广播允许 PyTorch 中<u>不同形状的 Tensor</u> 、<u>标量与 Tensor </u>进行计算。

**不同形状 Tensor 计算**：广播规则为

- 从两个 Tensor 的**末尾维度**开始，向前比较它们的维度。
- 如果两个维度相等，或者其中一个维度为 `1`​，那么它们是**兼容**的。
- 如果某个 Tensor 的维度比另一个少，它会被自动**添加**维度，并在前面补上 `1`​。

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # 形状: (2, 3)
b = torch.tensor([10, 20, 30])  # 形状: (3)

# PyTorch 会将 b 扩展为形状 (1, 3)，然后逐行复制以匹配 a 的形状 (2, 3)
# 最终相当于 a 和 [[10, 20, 30], [10, 20, 30]] 相加
c_broadcast = a + b
# c = tensor([[11, 22, 33],
#            [14, 25, 36]])
```

**标量与 Tensor 计算**：

```python
y = x * 2  # 标量 2 会被广播到 x 的所有元素
```

---

## 矩阵与向量运算

PyTorch 提供了专用的函数和运算符来支持矩阵乘法运算。

**二维矩阵计算**：两个二维张量（不支持矩阵和向量）

```python
# 假设 x、y 是可相乘的矩阵
z = torch.mm(x, y)
```

**批量矩阵计算**：对于三维 Tensor 可以将第一个维度视作批次大小（Batch Size），然后做批次乘法

```python
# 假设 batch_x 和 batch_y 是三维张量，希望对后两个维度做矩阵乘法
batch_z = torch.bmm(batch_x, batch_y)
# batch_z[i,:,:] = batch_x[i,:,:] @ batch_y[i,:,:]
```

**万能矩阵乘法**：`torch.matmul(a, b)`​ 或者 `a @ b`​ 能根据 Tensor 的维度采用最合适的操作：

- **向量 x 向量 (1D)** : 返回点积 (dot product)。
- **矩阵 x 向量 (2D @ 1D)** : 返回矩阵-向量乘积。
- **矩阵 x 矩阵 (2D @ 2D)** : 等同于 `torch.mm`​。
- **批量 x 批量 (3D+ @ 3D+)** : 支持广播机制的批量矩阵乘法。例如，一个 (J, 1, N, M) 的 Tensor 可以和 (K, M, P) 的 Tensor 相乘。

---

## 原地操作

PyTorch 中许多操作都有一个带下划线 `_`​ 的版本，这表示它们是原地（in-place）操作。它们会直接修改 Tensor 的内容，而不是创建一个新的 Tensor。

```python
# ============================================
# 基础操作：假设 x 和 y 是同 size 的 tensor，a 为 float
x.add_(y) # x = x + y
x.sub_(y) # x = x - y
x.mul_(a) # x 逐元素乘以 a
x.pow_(a) # x 逐元素取 a 次方
x.copy_(y) # 将 y 的内容复制到 x 中
# ============================================
# 高级操作：假设 x、y、z 是同 size 的 tensor，a 为 float
x.addcmul_(y, z, value=a) # x = x + a(y * z)，y 和 z 逐元素乘法，乘上 a，加到 x 上
x.addcdiv_(y, z, value=a) # x = x + a(y / z)，y 和 z 逐元素除法，乘上 a，加到 x 上
```

**注意事项**：原地操作会节省内存，但可能会在计算梯度时（例如在 `backward()`​ 调用中）带来问题，因为它会修改用于计算梯度的前向传播的数值。除非你确定不会影响梯度计算或者内存非常紧张，否则建议使用非原地操作 (`x = x + y`​)，这样代码的可读性和安全性更高。一般原地操作常用于编写优化器。
