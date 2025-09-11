+++

title = "Pytest + VSCode 快速上手"

date = "2025-09-11"

[taxonomies]

tags = ["Python", "Pytest"]

+++

Pytest 是 Python 中好用的自动化测试框架，本文同时讲解了如何在 VSCode 中调用。

---

## Pytest 快速入门

**安装 Pytest**：

```bash
# 使用 pip 安装
pip install pytest

# 或者使用 conda 安装
conda install pytest
```

**Pytest 中的测试**：Pytest 中测试文件必须命名为 `test_*.py`​ 或 `*_test.py`​，测试函数必须以 `test_`​ 开头。例如

```python
# test_calculation.py

# 这是一个我们要测试的业务函数
def add(a, b):
    return a + b

# 这是一个测试函数
def test_add():
    # 使用 Python 原生的 assert 关键字进行断言
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```

**从命令行运行**：最简单的运行方式是使用命令行，`cd`​ 到项目测试根目录，然后运行

```bash
pytest
```

Pytest 会自动扫描当前目录及子目录下所有符合规则的测试文件和函数，并执行它们。你会看到一份清晰的报告。

```text
========================= test session starts =========================
platform linux -- Python 3.10.4, pytest-7.1.2, pluggy-1.0.0
rootdir: /path/to/your/project
collected 1 item

test_calculation.py .                                           [100%]

========================== 1 passed in 0.01s ==========================
```

---

## 在 VSCode 中使用 Pytest

**配置 VS Code**：首先<u>安装 Python 扩展</u>，然后<u>点击 VS Code 左侧活动栏的“烧杯”图标</u>，进入测试面板。<u>点击“配置 Python 测试”，在弹出的菜单中选择 </u>​**<u>​`pytest`​</u>**​，然后选择你的测试所在的根目录。在 `settings.json`​ 中可以修改配置文件

```json
{
  "python-envs.defaultEnvManager": "ms-python.python:conda",
  "python-envs.defaultPackageManager": "ms-python.python:conda",
  "python-envs.pythonProjects": [],
  "python.testing.pytestArgs": [
    "test" // 或者指定你的测试目录，例如 "tests/"
  ],
  "python.testing.unittestEnabled": false,
  "python.testing.pytestEnabled": true,
  "python.testing.cwd": "${workspaceFolder}" // 解决导入问题
}
```

**测试工作流**：在测试面板中，我们可以看到所有测试的树状结构。每个测试用例、每个文件、每个类旁边都有【Run】、【Debug】按钮，可以随时单独运行。

![image](assets/image-20250911112131-e2suaw7.png)​

‍
