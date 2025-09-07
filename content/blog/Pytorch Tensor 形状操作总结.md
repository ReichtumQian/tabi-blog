+++

title = "Pytorch Tensor 形状操作总结"

date = "2025-09-07"

[taxonomies]

tags = ["PyTorch", "Python"]

+++

|方法|返回类型|描述|示例：`(3, 4, 5)`​|
| :----: | :--------: | :--------------------------: | :--------: |
|​`x.shape`​|​`torch.Size`​|以元组形式返回 Tensor 形状|​`torch.Size([3, 4, 5])`​|
|​`x.shape[n]`​|​`int`​|返回第 `n`​ 个维度的大小|​`x.shape[0] = 3`​|
|​`x.size()`​|​`torch.Size`​|和 `x.shape`​ 相同|​`torch.Size([3, 4, 5])`​|
|​`x.size(n)`​|​`int`​|返回第 `n`​ 个维度的大小|​`x.size(1) = 4`​|
|​`len(x)`​|​`int`​|返回第 `0`​ 个维度的大小|​`len(x) = 3`​|
|​`x.dim()`​|​`int`​|返回维度数量|​`x.dim() = 3`​|
|​`x.numel()`​|​`int`​|返回 Tensor 元素总数|​`x.numel() = 60`​|

‍
