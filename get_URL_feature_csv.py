
import numpy as np
import csv

input_file = r"C:\Users\13488\Desktop\大创用\数据\URLfeatures.csv"  # 替换为你的CSV文件路径
output_file = r"C:\Users\13488\Desktop\大创用\数据\features\URL_getfromcsv.npz"  # 输出保存的npz文件路径

# 读取CSV文件并构建二维数组
data = []
with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # 跳过第一行
    for index, row in enumerate(csvreader, start=1):
        # 跳过第一列和第二列，从第三列开始取数据
        cleaned_row = [index] + list(map(float, row[2:]))  # 转换为浮点数
        data.append(cleaned_row)

# 转换为NumPy数组
data_array = np.array(data)

# 保存为npz文件
np.savez(output_file, data=data_array)

print(f"数据已保存到文件: {output_file}")

# # 加载npz文件
# loaded_data = np.load(output_file)
# 
# # 从npz文件中获取数据数组
# data_array = loaded_data['data']
# 
# # 打印部分数据内容
# print(f"数据的形状：{data_array.shape}")
# print("前5行数据内容：")
# print(data_array[:5, :].shape)  # 打印前5行数据内容

