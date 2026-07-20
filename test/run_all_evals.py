# import os
# import subprocess
# import re

# # 基础目录路径
# BASE_DIR = r"C:\\Users\\BoMu\\Desktop\\gravityface\\实验数据\\totaltab\\Output"

# # 三个评估脚本的路径 (假设它们和本脚本在同一目录下)
# SCRIPT_IJBB = "eval_ijbb.py"
# SCRIPT_TINYFACE = "validate_tinyface.py"
# SCRIPT_VERIF = "verif.py"

# def main():
#     # 遍历总目录下的所有子文件夹
#     for folder_name in os.listdir(BASE_DIR):
#         folder_path = os.path.join(BASE_DIR, folder_name)
        
#         # 跳过非文件夹
#         if not os.path.isdir(folder_path):
#             continue

#         model_path = os.path.join(folder_path, "model.pt")
        
#         # 检查模型是否存在
#         if not os.path.exists(model_path):
#             print(f"⚠️ [跳过] 未在 {folder_path} 中找到 model.pt")
#             continue

#         network = "r50"

#         # 定义结果保存路径为该模型文件夹下的 eval_results 目录
#         result_dir = os.path.join(folder_path, "insightface")
#         os.makedirs(result_dir, exist_ok=True)

#         print(f"\n{'='*50}")
#         print(f"🚀 开始评估模型: {folder_name}")
#         print(f"🧠 网络结构: {network}")
#         print(f"📂 结果将保存至: {result_dir}")
#         print(f"{'='*50}\n")

#         # ---------------------------------------------------
#         # 1. 运行 IJBB
#         # ---------------------------------------------------
#         print(">>> 正在运行 IJBB 评估...")
#         cmd_ijbb = [
#             "python", SCRIPT_IJBB,
#             "--model_path", model_path,
#             "--result_dir", result_dir,
#             "--network", network
#         ]
#         subprocess.run(cmd_ijbb)

#         # ---------------------------------------------------
#         # 2. 运行 TinyFace
#         # ---------------------------------------------------
#         print("\n>>> 正在运行 TinyFace 评估...")
#         cmd_tinyface = [
#             "python", SCRIPT_TINYFACE,
#             "--model_path", model_path,
#             "--result_dir", result_dir,
#             "--network", network
#         ]
#         subprocess.run(cmd_tinyface)

#         # ---------------------------------------------------
#         # 3. 运行 Verif
#         # ---------------------------------------------------
#         print("\n>>> 正在运行 7大测试集 Verif 评估...")
#         cmd_verif = [
#             "python", SCRIPT_VERIF,
#             "--model_path", model_path,
#             "--result_dir", result_dir,
#             "--network", network
#         ]
#         subprocess.run(cmd_verif)

#     print("\n🎉 所有文件夹的评估任务已全部完成！")

# if __name__ == "__main__":
#     main()
import os
import subprocess
import re

# 基础目录路径
BASE_DIR = r"D:\FRcode\yingli\test_all\Output"

# 三个评估脚本的路径 (假设它们和本脚本在同一目录下)
SCRIPT_IJBB = "eval_ijbb.py"
SCRIPT_IJBC = "eval_ijbc.py"
SCRIPT_TINYFACE = "validate_tinyface.py"
SCRIPT_VERIF = "verif.py"

def main():
    # 遍历总目录下的所有子文件夹
    for folder_name in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        # 跳过非文件夹
        if not os.path.isdir(folder_path):
            continue

        model_path = os.path.join(folder_path, "model.pt")
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            print(f"⚠️ [跳过] 未在 {folder_path} 中找到 model.pt")
            continue

        # ====================================================
        # 新增：使用正则表达式从文件夹名中提取网络结构 (r50 或 r100)
        # ====================================================
        match = re.search(r"(r50|r100)", folder_name.lower())
        if match:
            network = match.group(1)
        else:
            print(f"⚠️ [跳过] 无法从文件夹名 {folder_name} 中识别网络结构(缺少 r50 或 r100)")
            # 如果你想设置默认值而不是跳过，可以注释掉上面两行并取消下方注释：
            # network = "r50" 
            # print(f"⚠️ [警告] 无法识别网络结构，默认使用 r50")
            continue

        # 定义结果保存路径为该模型文件夹下的 insightface 目录
        result_dir = os.path.join(folder_path, "EVAL")
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"🚀 开始评估模型: {folder_name}")
        print(f"🧠 网络结构: {network}")
        print(f"📂 结果将保存至: {result_dir}")
        print(f"{'='*50}\n")

        # ---------------------------------------------------
        # 1. 运行 IJBB
        # ---------------------------------------------------
        print(">>> 正在运行 IJBB 评估...")
        cmd_ijbb = [
            "python", SCRIPT_IJBB,
            "--model_path", model_path,
            "--result_dir", result_dir,
            "--network", network
        ]
        subprocess.run(cmd_ijbb)

        # ---------------------------------------------------
        # 2. Run IJBC
        # ---------------------------------------------------
        print("\n>>> Running IJBC evaluation...")
        cmd_ijbc = [
            "python", SCRIPT_IJBC,
            "--model_path", model_path,
            "--result_dir", result_dir,
            "--network", network
        ]
        subprocess.run(cmd_ijbc)

        # ---------------------------------------------------
        # 2. 运行 TinyFace
        # ---------------------------------------------------
        print("\n>>> 正在运行 TinyFace 评估...")
        cmd_tinyface = [
            "python", SCRIPT_TINYFACE,
            "--model_path", model_path,
            "--result_dir", result_dir,
            "--network", network
        ]
        subprocess.run(cmd_tinyface)

        # ---------------------------------------------------
        # 3. 运行 Verif
        # ---------------------------------------------------
        print("\n>>> 正在运行 7大测试集 Verif 评估...")
        cmd_verif = [
            "python", SCRIPT_VERIF,
            "--model_path", model_path,
            "--result_dir", result_dir,
            "--network", network
        ]
        subprocess.run(cmd_verif)

    print("\n🎉 所有文件夹的评估任务已全部完成！")

if __name__ == "__main__":
    main()
