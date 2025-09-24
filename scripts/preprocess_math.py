import os
import re
import sys

def escape_math_content(delimiter, content):
    """
    转义数学公式内容中的特殊 Markdown 字符。
    这个函数只处理传入的公式内容字符串。
    """
    processed_content = content

    # 处理行首的减号，防止被解析为列表（仅对块级公式 $$）
    if delimiter == '$$':
        processed_content = re.sub(r'^\s*-', r'\\-', processed_content, flags=re.MULTILINE)
    
    # 处理下划线：将不是由反斜杠开头的 `_` 替换为 `\_`
    processed_content = re.sub(r'(?<!\\)_', r'\\_', processed_content)
    
    # 处理星号：将不是由反斜杠开头的 `*` 替换为 `\*`
    processed_content = re.sub(r'(?<!\\)\*', r'\\*', processed_content)
    
    # 处理 LaTeX 换行符：将 `\\` 替换为 `\\\\`
    processed_content = re.sub(r'\\\\', r'\\\\\\\\', processed_content)
    
    # 处理 LaTeX 双竖线：将 `\|` 替换为 `\\|`
    processed_content = re.sub(r'\\\|', r'\\\\|', processed_content)
    
    # 处理大括号：将 `\{` 和 `\}` 分别替换为 `\\{` 和 `\\}`
    processed_content = re.sub(r'\\{', r'\\\\{', processed_content)
    processed_content = re.sub(r'\\}', r'\\\\}', processed_content)

    # 将 `\%` 替换为 `\\%`
    processed_content = re.sub(r'\\%', r'\\\\%', processed_content)
    
    return processed_content

def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，通过状态解析安全地处理数学公式中的
    特殊字符，以避免与 Markdown 语法冲突。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # --- 第 1 步：保护已转义的美元符号 ---
        escaped_dollars = []
        def store_escaped_dollar(match):
            escaped_dollars.append(match.group(0))
            return f"__ESCAPED_DOLLAR_{len(escaped_dollars) - 1}__"
        
        content_no_escapes = re.sub(r'\\(\$\$|\$)', store_escaped_dollar, original_content)

        # --- 第 2 步：保护所有代码块 ---
        code_blocks = []
        def store_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        code_pattern = re.compile(r'```.*?```|`.*?`', re.DOTALL)
        content_no_code = code_pattern.sub(store_code_block, content_no_escapes)

        # --- 第 3 步：通过手动解析安全地处理数学公式 ---
        processed_parts = []
        last_index = 0
        # 匹配所有可能的分隔符，顺序很重要，优先匹配 `$$`
        delimiter_pattern = re.compile(r'\$\$|\$|~')
        
        in_math_block = False
        block_delimiter = None
        block_start_index = 0

        for match in delimiter_pattern.finditer(content_no_code):
            delimiter = match.group(0)
            current_pos = match.start()

            if not in_math_block:
                # 发现新的公式块起始符
                in_math_block = True
                block_delimiter = delimiter
                # 先把上一个公式块到这一个块之间的普通文本添加进来
                processed_parts.append(content_no_code[last_index:current_pos])
                block_start_index = match.end()
            elif delimiter == block_delimiter:
                # 发现匹配的公式块结束符
                math_content = content_no_code[block_start_index:current_pos]
                processed_math = escape_math_content(delimiter, math_content)
                
                # 拼接完整的、处理过的公式块
                processed_parts.append(f"{delimiter}{processed_math}{delimiter}")
                
                # 重置状态
                in_math_block = False
                last_index = match.end()
        
        # 添加最后一个公式块之后剩余的文本
        if last_index < len(content_no_code):
            processed_parts.append(content_no_code[last_index:])

        processed_content = "".join(processed_parts)

        # --- 第 4 步：按相反顺序恢复内容 ---
        # 恢复代码块
        for i, block in reversed(list(enumerate(code_blocks))):
            processed_content = processed_content.replace(f"__CODE_BLOCK_{i}__", block, 1)
        
        # 恢复已转义的美元符号
        for i, dollar in reversed(list(enumerate(escaped_dollars))):
            processed_content = processed_content.replace(f"__ESCAPED_DOLLAR_{i}__", dollar, 1)

        # --- 最后：如果内容有变化，则写回文件 ---
        if processed_content != original_content:
            print(f'  -> Modifying: {filepath}')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_content)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def main(directory):
    """
    遍历指定目录下的所有 .md 文件并进行处理。
    """
    print(f"Starting math pre-processing in directory: '{directory}'...")
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                process_markdown_file(os.path.join(root, file))
    
    print("Processing complete.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        content_dir = sys.argv[1]
    else:
        # 如果没有提供路径，则默认为 'content' 目录
        content_dir = 'content' 
    
    main(content_dir)

