import os

def export_dir_structure(root_dir, output_txt):
    """
    导出文件夹目录结构到TXT文件（修复sub_prefix未定义问题，适配Linux/Windows）
    :param root_dir: 要遍历的根文件夹路径（绝对/相对路径均可）
    :param output_txt: 输出的TXT文件路径（如：./目录结构.txt）
    """
    # 校验根文件夹是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：文件夹 {root_dir} 不存在！")
        return

    # 以utf-8编码打开TXT（避免中文乱码），覆盖写入
    with open(output_txt, 'w', encoding='utf-8') as f:
        # 写入根文件夹名称（作为标题）
        root_name = os.path.basename(root_dir) or root_dir  # 处理/根路径的情况
        f.write(f"┌── {root_name}\n")

        # 遍历文件夹：os.walk会递归遍历所有子文件夹，topdown=True先遍历上层
        for parent_path, sub_dirs, files in os.walk(root_dir):
            # 计算当前路径相对根目录的层级（根目录是0级，子文件夹1级，以此类推）
            level = parent_path.replace(root_dir, '').count(os.sep)
            # 生成层级前缀（控制缩进，非最后一级用│ 占位，保证树形对齐）
            prefix = '│  ' * level
            # 【核心修复】提前初始化sub_prefix，所有场景都能使用，避免未定义
            sub_prefix = '│  ' * (level + 1)
            # 当前文件夹的名称
            current_dir_name = os.path.basename(parent_path)
            # 写入当前文件夹（根文件夹已写，跳过）
            if parent_path != root_dir:
                f.write(f"{prefix}├── {current_dir_name}\n")

            # 处理当前文件夹下的所有文件（根目录/子目录都能正常执行）
            for idx, file_name in enumerate(files):
                # 判断是否是最后一个文件（最后一个用└─，其他用├─）
                # 最后一个文件：是文件列表最后一个 + 当前文件夹无剩余子目录
                is_last_file = idx == len(files) - 1 and len(sub_dirs) == 0
                if is_last_file:
                    f.write(f"{sub_prefix}└── {file_name}\n")
                else:
                    f.write(f"{sub_prefix}├── {file_name}\n")

    print(f"目录结构已成功导出到：{os.path.abspath(output_txt)}")

# ------------------- 适配你的Linux环境，修改这两个路径即可 -------------------
if __name__ == "__main__":
    # 要遍历的目标文件夹（你的Linux路径，示例：/data 或 /data/zjy_work）
    TARGET_FOLDER = r"/data/zjy_work/BGDiff"  # Linux路径直接写，r前缀不影响，可保留
    # 输出的TXT文件路径（示例：/data/目录结构.txt）
    OUTPUT_TXT = r"/data/zjy_work/BGDiff/tree.txt"
    # 执行导出
    export_dir_structure(TARGET_FOLDER, OUTPUT_TXT)