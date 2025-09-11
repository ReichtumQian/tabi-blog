+++

title = "Pytest 中的 Fixture"

date = "2025-09-11"

[taxonomies]

tags = ["Python"]

+++

Pytest 是 Python 中的一个单元测试框架，而 Fixture 是为 Pytest 测试提供其运行所需的数据、对象、环境和状态的函数。它负责“准备工作”（Setup）和“清理工作”（Teardown）。

---

## Pytest Fixture 的基础用法

**什么是 Fixture**：在 Pytest 中，一个 Fixture 就是一个被 `@pytest.fixture`​ 装饰器标记的 Python 函数。这个函数的名字可以被其他测试函数当作参数来使用。当 Pytest 发现一个测试函数请求了一个 Fixture，它会：

1. 在运行该测试函数**之前**，先去执行对应的 Fixture 函数。
2. 将 Fixture 函数的返回值（如果有的话）传递给测试函数。
3. 在测试函数**执行完毕后**，执行 Fixture 中定义的清理操作。

 **@pytest.fixture 的用法**：假设我们有好几个测试用例，都需要用到一个相同的数据字典。

- 没有 Fixture 的写法（不推荐）：`user_data`​ 这个字典在每个测试中都重复定义了，非常冗余。

```python
# test_without_fixture.py

def test_user_info_display():
    user_data = {"name": "Alice", "email": "alice@example.com", "role": "user"}
    # ... 测试显示逻辑 ...
    assert user_data["name"] == "Alice"

def test_user_permission():
    user_data = {"name": "Alice", "email": "alice@example.com", "role": "user"}
    # ... 测试权限逻辑 ...
    assert user_data["role"] == "user"
```

- 使用 Fixture 的写法（推荐）：Pytest 在执行 `test_user_info_display`​ 和 `test_user_permission`​ 时，会自动找到并执行 `user_data`​ Fixture，然后将它的返回值 `data`​ 注入到测试函数的 `user_data`​ 参数中。

```python
# test_with_fixture.py
import pytest

# 1. 使用 @pytest.fixture 装饰器来定义一个 Fixture
@pytest.fixture
def user_data():
    """这是一个提供用户数据的 Fixture"""
    print("\n--- (Setup) 创建 user_data ---")
    data = {"name": "Alice", "email": "alice@example.com", "role": "user"}
    return data

# 2. 测试函数直接把 Fixture 函数名作为参数
def test_user_info_display(user_data):
    """测试用户信息显示"""
    print("--- (Test) 正在运行 test_user_info_display ---")
    assert user_data["name"] == "Alice"

def test_user_permission(user_data):
    """测试用户权限"""
    print("--- (Test) 正在运行 test_user_permission ---")
    assert user_data["role"] == "user"
```

**使用 yield 实现 Setup 和 Teardown**：如果我们需要清理工作（Teardown），我们可以使用 `yield`​ 关键字，其之前的代码是 Setup 部分，其之后的代码是 Teardown 部分。

```python
# test_with_teardown.py
import pytest
import os

@pytest.fixture
def temp_file():
    """创建一个临时文件，测试后删除"""
    # --- Setup: yield 之前的部分 ---
    print("\n--- (Setup) 创建临时文件 temp.txt ---")
    file_path = "temp.txt"
    with open(file_path, "w") as f:
        f.write("hello pytest")

    # 使用 yield 将文件路径提供给测试函数
    yield file_path

    # --- Teardown: yield 之后的部分 ---
    print("\n--- (Teardown) 删除临时文件 temp.txt ---")
    os.remove(file_path)

def test_read_temp_file(temp_file):
    """测试读取临时文件的内容"""
    print(f"--- (Test) 正在读取文件: {temp_file} ---")
    with open(temp_file, "r") as f:
        content = f.read()
    assert content == "hello pytest"

def test_file_exists(temp_file):
    """测试文件是否存在"""
    print(f"--- (Test) 正在检查文件是否存在: {temp_file} ---")
    assert os.path.exists(temp_file)
```

---

## Fixture 进阶用法

**Fixture 的 Scope**：默认情况下，Fixture 的作用域是 `function`​，意味着它会对**每一个**使用它的测试函数都执行一次，有时候这会造成不必要的开销。我们可以通过 `scope`​ 参数来控制 Fixture 的生命周期：

- `scope="function"`​ (默认): 每个测试函数执行一次。
- `scope="class"`: 每个测试类（Test Class）只执行一次。
- `scope="module"`: 每个模块（`.py`​ 文件）只执行一次。
- `scope="session"`: 整个测试会话（运行 `pytest`​ 命令一次）只执行一次。

例如连接数据库是一个耗时操作，我们希望整个测试文件（模块）只连接一次：

```python
# test_db_connection.py
import pytest

# 使用 scope="module"
@pytest.fixture(scope="module")
def db_conn():
    """一个昂贵的数据库连接，整个模块共享"""
    print("\n--- (Setup) 建立昂贵的数据库连接 (scope=module) ---")
    # 模拟一个连接对象
    conn = {"status": "connected"}
    yield conn
    print("\n--- (Teardown) 关闭数据库连接 (scope=module) ---")
    conn["status"] = "closed"

class TestUser:
    def test_add_user(self, db_conn):
        print("--- (Test) 测试添加用户 ---")
        assert db_conn["status"] == "connected"

    def test_delete_user(self, db_conn):
        print("--- (Test) 测试删除用户 ---")
        assert db_conn["status"] == "connected"
```

**Fixture 之间相互依赖**：一个 Fixture 可以像测试函数一样，请求另一个 Fixture

```python
# test_fixture_dependency.py
import pytest

@pytest.fixture(scope="session")
def api_server():
    """启动一个 API 服务器，整个会话只启动一次"""
    print("\n--- (Setup) 启动 API 服务器 ---")
    server = {"url": "http://api.example.com"}
    yield server
    print("\n--- (Teardown) 关闭 API 服务器 ---")

@pytest.fixture
def api_client(api_server): # <--- 这里依赖了 api_server Fixture
    """创建一个 API 客户端，连接到服务器"""
    print("\n--- (Setup) 创建 API 客户端 ---")
    client = {"server_url": api_server["url"]}
    return client

def test_login(api_client):
    """测试登录接口"""
    print("--- (Test) 测试登录 ---")
    assert api_client["server_url"] == "http://api.example.com"
```
