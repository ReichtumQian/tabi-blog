import os
import re
import sys

def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，处理非代码块、非转义美元区域中数学公式的
    特殊字符，以避免与 Markdown 语法冲突。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # --- 第 1 步：保护已转义的美元符号 ---
        # 存储 `\$` 或 `\$\$`，防止它们被误认为是数学公式分隔符。
        escaped_dollars = []
        def store_escaped_dollar(match):
            escaped_dollars.append(match.group(0))
            return f"__ESCAPED_DOLLAR_{len(escaped_dollars) - 1}__"
        
        content_no_escapes = re.sub(r'\\(\$\$|\$)', store_escaped_dollar, original_content)

        # --- 第 2 步：保护所有代码块 ---
        # 存储 ` ```...``` 和 ` `...` ` 中的内容，防止其中的字符被错误处理。
        code_blocks = []
        def store_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        code_pattern = re.compile(r'```.*?```|`.*?`', re.DOTALL)
        content_no_code = code_pattern.sub(store_code_block, content_no_escapes)

        # --- 第 3 步：在清理后的文本上安全地处理数学公式 ---
        def escape_special_chars_in_math(match):
            """
            转义数学公式内容中的特殊 Markdown 字符。
            """
            start_delim, content, end_delim = match.groups()
            
            processed_content = content

            # 处理行首的减号，防止被解析为列表
            if start_delim == '$$':
                processed_content = re.sub(r'^\s*-', r'\\-', processed_content, flags=re.MULTILINE)
            
            # 处理下划线：将不是由反斜杠开头的 `_` 替换为 `\_`
            processed_content = re.sub(r'(?<!\\)_', r'\\_', processed_content)
            
            # 处理星号：将不是由反斜杠开头的 `*` 替换为 `\*`
            processed_content = re.sub(r'(?<!\\)\*', r'\\*', processed_content)

            # --- 新增逻辑：处理波浪号 ---
            # 将不是由反斜杠开头的 `~` 替换为 `\~`
            # 这样可以防止 Zola 将 `~...~` 解析为删除线。
            processed_content = re.sub(r'(?<!\\)~', r'\\~', processed_content)
            
            # 处理 LaTeX 换行符：将 `\\` 替换为 `\\\\`。
            # Markdown 会将 `\\\\` 解析为 `\\`，KaTeX 才能正确识别为换行。
            processed_content = re.sub(r'\\\\', r'\\\\\\\\', processed_content)
            
            # 处理 LaTeX 双竖线：将 `\|` 替换为 `\\|`。
            # Markdown 会将 `\\|` 解析为 `\|`，KaTeX 才能正确识别。
            # 这个替换必须在处理 `\\` 之后进行，以避免冲突。
            processed_content = re.sub(r'\\\|', r'\\\\|', processed_content)
            
            # 处理大括号
            # 将 `\{` 和 `\}` 分别替换为 `\\{` 和 `\\}`
            # Zola/Markdown 会将 `\\{` 解析为 `\{`，KaTeX 才能正确识别。
            processed_content = re.sub(r'\\{', r'\\\\{', processed_content)
            processed_content = re.sub(r'\\}', r'\\\\}', processed_content)

            # 将 `\%` 替换为 `\\%`
            # Zola/Markdown 会将 `\\%` 解析为 `\%`，KaTeX 才能正确识别。
            processed_content = re.sub(r'\\%', r'\\\\%', processed_content)
            
            return f"{start_delim}{processed_content}{end_delim}"

        # 使用非贪婪匹配来查找由 $...$ 或 $$...$$ 包围的数学公式
        math_pattern = re.compile(r'(\$\$|\$)(.*?)(\1)', re.DOTALL)
        processed_content = math_pattern.sub(escape_special_chars_in_math, content_no_code)

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

