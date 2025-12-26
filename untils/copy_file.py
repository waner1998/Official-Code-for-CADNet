import os
import shutil
            
def copy_file(destination_folder_path):
    current_folder = os.getcwd()

    files = os.listdir(current_folder)
    for file in files:
        if file.endswith(".py") and os.path.isfile(file):
            shutil.copy(file, destination_folder_path)
            
    for specified_folder_name in ['dataloader', 'model', 'untils', ]:
        source_folder = os.path.join(current_folder, specified_folder_name)

        # 检查指定文件夹是否存在
        if os.path.exists(source_folder) and os.path.isdir(source_folder):
            # 构建目标文件夹路径
            destination_path = os.path.join(destination_folder_path, specified_folder_name)

            # 复制文件夹及其内容到目标文件夹
            shutil.copytree(source_folder, destination_path)
        else:
            print(f"指定的文件夹 '{specified_folder_name}' 不存在或不是一个文件夹。")

