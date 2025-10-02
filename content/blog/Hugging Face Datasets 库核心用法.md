+++

title = "Hugging Face Datasets 库核心用法"

date = "2025-10-02"

[taxonomies]

tags = ["Python", "Datasets Library"]

+++

`datasets`​ 是 Hugging Face 提供的一站式数据集管理库，其核心功能包含：

- 数据的加载与创建
- 数据的查看与索引
- 数据的核心处理方法
- 数据的分割与合并

---

## 数据的加载与创建

**load_dataset 加载 HuggingFace Hub 数据集**：`load_dataset`​ 是 `datasets`​ 库的核心操作，其从各种来源加载数据集。一般情况下返回一个 `DatasetDict`​ 对象，包含了多个数据分割（例如 `train`​、`test`​、`validation`​）。使用 `dataset_dict["train"]`​ 获取对应的 `Dataset`​

```python
from datasets import load_dataset

# 示例1: 加载 SQuAD 数据集 (只有一个默认配置)
# 这会下载并加载所有可用的数据分割 (splits)，比如 'train' 和 'validation'
dataset_dict = load_dataset("squad")

print(dataset_dict)
# 输出:
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers'],
#         num_rows: 87599
#     })
#     validation: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers'],
#         num_rows: 10570
#     })
# })

# 可以像字典一样访问特定的数据分割
train_dataset = dataset_dict["train"]
print(train_dataset[0]) # 查看训练集的第一条数据
```

**load_dataset 的参数**：`load_dataset`​ 有以下核心参数：

- `path`​：必需，可以是 Hub 上数据集的 ID，也可以是本地脚本的路径
- `name`​：可选，用于指定数据集的配置或者子集。例如 `GLUE`​ 数据集包含了多个子任务（`mrpc`​、`qqp`​ 等），需要指定一个 `name`​
- `split`​：可选，直接加载一个或多个数据分割，如果只加载一个分割，则返回 `Dataset`​ 对象，而非 `DatasetDict`​ 对象
- `streaming=True`​：可选，可以从 Hugging Face 流式获取数据

**from_dict 直接从内存加载**：使用 `from_dict`​ 可以从内存直接加载数据集

```python
from datasets import Dataset

my_data = {
    "id": [1, 2, 3],
    "text": ["Hugging Face is great!", "I love NLP.", "This is a sentence."],
    "label": [1, 1, 0]
}

dataset = Dataset.from_dict(my_data)
print(dataset)
# 输出:
# Dataset({
#     features: ['id', 'text', 'label'],
#     num_rows: 3
# })
```

---

## 数据的查看与索引

**像列表一样操作**：`DataSet`​ 对象可以使用 `len`​、`[0]`​、`[0:2]`​ 等索引

```python
# 获取数据集大小
print(len(dataset))  # 输出: 3

# 获取第一条数据
first_item = dataset[0]
print(first_item)
# 输出 (一个字典):
# {'id': 1, 'text': 'Hugging Face is great!', 'label': 1}

# 获取一个切片 (返回一个新的 Dataset)
subset = dataset[0:2]
print(subset)
# Dataset({
#     features: ['id', 'text', 'label'],
#     num_rows: 2
# })
```

**像字典一样操作**：可以通过列名访问整列数据

```python
# 获取所有 'text' 列的数据
all_texts = dataset['text']
print(all_texts)
# 输出 (一个列表):
# ['Hugging Face is great!', 'I love NLP.', 'This is a sentence.']
```

**查看数据结构**：

```python
# 查看列名
print(dataset.column_names)  # ['id', 'text', 'label']

# 查看特征 (包含列名和数据类型)
print(dataset.features)
# {'id': Value(dtype='int64', id=None), 'text': Value(dtype='string', id=None), 'label': Value(dtype='int64', id=None)}
```

---

## 数据的核心处理方法

**map 方法**：`datasets`​ 库使用 `.map`​ 方法将一个函数应用到数据集的每个样本上，常用于分词（Tokenization）、数据清洗、数据增强等。

- `batched=True`​：将样本打包为批次进行处理
- `num_proc`​：指定 CPU 个数
- `remove_columns`​：在 `map`​ 后移除不再需要的原始列，例如 `remove_columns=["text"]`​

```python
from transformers import AutoTokenizer

# 假设我们有一个 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义一个处理函数
def tokenize_function(examples):
    # examples 是一个字典，key是列名，value是数据的列表
    # 例如: {'text': ['Hugging Face is great!', 'I love NLP.', ...]}
    # tokenizer 可以直接处理列表输入
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# batched=True 让函数一次性接收一批数据 (a batch of examples)
# 这极大地提高了 tokenizer 的处理速度
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset)
# Dataset({
#     features: ['id', 'text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
#     num_rows: 3
# })
```

**filter 方法进行筛选**：`filter()`​ 根据一个返回布尔值的函数来筛选样本。

```python
# 筛选出文本长度大于 15 的样本
long_text_dataset = dataset.filter(lambda example: len(example['text']) > 15)
print(len(long_text_dataset)) # 输出: 1
```

**shuffle 方法进行打乱**：

```python
# seed 参数确保每次打乱的结果都一样，保证实验可复现
shuffled_dataset = dataset.shuffle(seed=42)
print(shuffled_dataset[0]['text'])
```

---

## 数据分割与合并

**train_test_split 分割数据**：如果你的数据集只有一个分割，可以用这个函数轻松地切分出训练集和测试集。

```python
# 切分出 20% 的数据作为测试集
train_test_dict = dataset.train_test_split(test_size=0.2)
print(train_test_dict)
# DatasetDict({
#     train: Dataset(...)
#     test: Dataset(...)
# })

train_split = train_test_dict['train']
test_split = train_test_dict['test']
```

**concatenate_datasets 合并数据集**：将多个 `Dataset`​ 对象合并成一个。

```python
from datasets import concatenate_datasets

dataset1 = Dataset.from_dict({"a": [1, 2]})
dataset2 = Dataset.from_dict({"a": [3, 4]})

combined_dataset = concatenate_datasets([dataset1, dataset2])
print(len(combined_dataset)) # 输出: 4
```

‍
