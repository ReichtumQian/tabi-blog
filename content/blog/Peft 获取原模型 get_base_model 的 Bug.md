+++

title = "Peft 获取原模型 get_base_model 的 Bug"

date = "2025-09-22"

[taxonomies]

tags = ["PyTorch", "Peft Library"]

+++

今天在 `peft`​ 库上折腾了一天，发现 `peft`​ 库的 `get_base_model()`​不能返回原始模型，还是返回了微调后的模型，因此记录一下。

> 参考 GitHub Issue：[`get_base_model()` is returning the base model with the LoRA still applied. · Issue #430 · huggingface/peft](https://github.com/huggingface/peft/issues/430)

`get_base_model()`​ **的错误使用**：当前不要用 `get_base_model()`​ 去获取原始模型。例如我们可能如下使用：

```python
self.model = peft.get_peft_model(self.model, lora_config)
self.reference_model = self.model.get_base_model()
self.reference_model.eval()
```

我们以为 `reference_model`​ 是未经过微调的模型，实际上 `peft`​ 返回了经过微调的模型，也就是 `reference_model`​ 和 `model`​ 没有差别！

**如何正确获取未经微调的模型**：建议使用 `with model.disable_adapter()`​，例如

```python
with model.disable_adapter():
	model(inputs)
```
