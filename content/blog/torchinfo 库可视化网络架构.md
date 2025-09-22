+++

title = "torchinfo 库可视化网络架构"

date = "2025-09-22"

[taxonomies]

tags = ["PyTorch", "Python"]

+++



[TylerYep/torchinfo](https://github.com/TylerYep/torchinfo) 库提供了便捷、美观的网络架构输出方案

---

## 安装 torchinfo

```bash
# 使用 pip
pip install torchinfo
# 使用 conda
conda install -c conda-forge torchinfo
```

---

## 快速上手

为了方便检查每一层的 `output shape`​，`torchinfo`​ 要求输入 `input size`​，一般情况下不在乎 `batch_size`​ 的话可以设置为 `1`​。

```python
from torchinfo import summary

model = ConvNet()
batch_size = 16
summary(model, input_size=(batch_size, 1, 28, 28))
# 注意在 jupyter notebook 中 summary 需要加上 print 才能正常输出
# print(summary(model, input_size=(batch_size, 1, 28, 28)))
```

输出如下

```bash
================================================================================================================
Layer (type:depth-idx)          Input Shape          Output Shape         Param #            Mult-Adds
================================================================================================================
SingleInputNet                  [7, 1, 28, 28]       [7, 10]              --                 --
├─Conv2d: 1-1                   [7, 1, 28, 28]       [7, 10, 24, 24]      260                1,048,320
├─Conv2d: 1-2                   [7, 10, 12, 12]      [7, 20, 8, 8]        5,020              2,248,960
├─Dropout2d: 1-3                [7, 20, 8, 8]        [7, 20, 8, 8]        --                 --
├─Linear: 1-4                   [7, 320]             [7, 50]              16,050             112,350
├─Linear: 1-5                   [7, 50]              [7, 10]              510                3,570
================================================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
Total mult-adds (M): 3.41
================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 0.40
Params size (MB): 0.09
Estimated Total Size (MB): 0.51
================================================================================================================
```
