
def extract_section(filename, start_string, end_string):
    # 用于标记是否已经开始提取片段
    in_section = False
    # 存储提取的片段内容
    section_lines = []

    # 打开文件并逐行读取
    with open(filename, 'r') as file:
        for line in file:
            # 如果已经开始提取片段，检查是否遇到结束字符串
            if in_section:
                print(line)
                if end_string in line:
                    break
            # 如果遇到开始字符串，开始提取片段
            elif start_string in line:
                in_section = True
                print(line)
    
    # 返回提取的片段内容
    return ''.join(section_lines)