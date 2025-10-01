+++

title = "lm_eval 库简介"

date = "2025-10-01"

[taxonomies]

tags = ["LLM", "Evaluation"]

+++

`lm_eval`​ (Language Model Evaluation Harness) 是一个开源的 Python 库，专门用于评估语言模型（LMs）。这里 `Harness`​ 表示【评估框架】。

---

## lm_eval 库介绍

**核心思想**：`lm_eval`​ 库的核心思想是<u>将模型、任务、评估过程解耦</u>。

- 模型：支持 Hugging Face `transformers`​ 库加载的模型、通过 API 调用的模型等
- 任务：各种类型的任务，例如【常识】、【阅读理解】、【数学能力】、【代码能力】、【知识问答】等
- 评估：`lm_eval`​ 负责将指定的模型在指定的任务上进行测试，计算出模型的得分

---

## simple_evaluate 函数

**simple_evaluate 函数**：`simple_evaluate`​ 使得评估过程变得简单，其参数如下

```python
from lm_eval import simple_evaluate

results = simple_evaluate(
    model,              # 要评估的模型对象或模型名称
    tasks,              # 一个包含任务名称的列表
    num_fewshot,        # 提供给模型的" few-shot"示例数量
    batch_size,         # 批处理大小，一次给模型多少个问题
    device,             # 指定在哪个设备上运行 (e.g., "cuda:0", "cpu")
    limit,              # 可选，限制每个任务测试的样本数量，用于快速测试
    ...                 # 其他一些高级配置
)
```

- `model`​：`"hf-auto"`​ 或 `"hf"`​ 表示使用 Hugging Face `transformers`​ 库加载，必须配合 `model_args="pretrained=MODEL_NAME_OR_PATH,peft=LORA_PATH"`​ 的；`vllm`​ 表示使用 `vLLM`​ 框架；`openai`​ 表示使用 OpenAI 的 API
- `tasks`​：Task 名称，具体可以去 [lm-evaluation-harness/lm_eval/tasks at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) 找
- `num_fewshot`​：整数如 `0`​、`5`​、`10`​ 等，在正式提问前给模型看几个例子
- `batch_size`​：整数，批次大小
- `device`​：`cuda`​、`cuda:0`​、`cpu`​ 等，告诉程序在哪推理
- `limit`​：限制样本数

**simple_evaluate 返回值**：`simple_evaluate`​ 会返回一个字典，内部包含详细的评估结果

```python
{
  "results": {
    "task_name_1": {
      "metric_1": score_1,
      "metric_2": score_2,
      ...
    },
    "task_name_2": {
      "metric_A": score_A,
      ...
    }
  },
  "versions": { ... },
  "config": { ... }
}
```
