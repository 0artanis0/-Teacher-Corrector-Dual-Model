# -*-coding:utf-8-*-
"""
将分类好的图片按照正确和错误保存分别保存到数据集文件夹下。便于直观的观察分类情况

    @Project: Dataset_label_error_cleanup_classifier
    @File   : ImgCopy.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import shutil
from urllib.parse import unquote


class CustomError(Exception):
    pass


class ImgCopy(object):
    """用于把切片的图片分类到两个文件夹，便于观察

    dataset_path 输入数据集路径，数据集文件夹格式必须为VOC

    output_type只可设置为false或者right"""

    def __init__(self, dataset_path, output_type):
        if output_type.lower() == 'false':
            print()
        elif output_type.lower() == 'right':
            print()
        else:
            raise CustomError('output_type mast be false or right')

        self.dataset_path = dataset_path
        self.file_path = dataset_path + '/CroppedImages/' + output_type + '.txt'
        self.source_folder = dataset_path + '/CroppedImages'
        self.destination_folder = dataset_path + '/' + output_type

        # 检查输出路径是否存在，不存在则提前创建
        with open(self.file_path, 'r') as file:
            self.lines = file.readlines()

    def copy_images_from_list(self):
        """根据"""
        file_list = self.lines
        source_folder = self.source_folder
        destination_folder = self.destination_folder

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for line in file_list:
            elements = line.strip().split()
            if len(elements) <= 2:
                image_name = elements[0]  # 解码文件名
                source_path = os.path.join(source_folder, image_name)
                destination_path = os.path.join(destination_folder, image_name)

                try:
                    shutil.copy2(source_path, destination_path)
                    print(f"文件 '{image_name}' 已拷贝到目标文件夹 '{destination_folder}'")
                except Exception as e:
                    print(f"拷贝文件 '{image_name}' 到目标文件夹 '{destination_folder}' 时发生错误: {e}")
            else:
                print(f"忽略行 '{line.strip()}'，因为缺少所需的元素")


dataset_path1 = 'D:/Paper with code/Trials/datasetB15/VOC2007'
dataset_path2 = 'D:/Paper with code/Trials/datasetB15/VOC2012'
# dataset_path = 'C:/Users/zhl98/Desktop/VOC2007'

if __name__ == '__main__':
    img_right1 = ImgCopy(dataset_path1, output_type='right')
    img_false1 = ImgCopy(dataset_path1, output_type='false')
    img_right1.copy_images_from_list()
    img_false1.copy_images_from_list()

    img_right2 = ImgCopy(dataset_path2, output_type='right')
    img_false2 = ImgCopy(dataset_path2, output_type='false')
    img_right2.copy_images_from_list()
    img_false2.copy_images_from_list()

    # img_right = ImgCopy(dataset_path, output_type='right')
    # img_false = ImgCopy(dataset_path, output_type='false')
    # img_right.copy_images_from_list()
    # img_false.copy_images_from_list()

# # 读取错误分类文件名列表
# file_path = 'C:/Users/zhl98/Desktop/VOC2023/CroppedImages/false.txt'  # 替换为你的txt文件路径
# with open(file_path, 'r') as file:
#     lines = file.readlines()
#
# # 指定源文件夹和错误分类目标文件夹
# source_folder = 'C:/Users/zhl98/Desktop/VOC2023/CroppedImages'  # 替换为你的源文件夹路径
# destination_folder = 'C:/Users/zhl98/Desktop/VOC2023/false'  # 替换为你的目标文件夹路径
#
# # 读取正确分类文件名列表
# file_path2 = 'C:/Users/zhl98/Desktop/VOC2023/CroppedImages/right.txt'  # 替换为你的txt文件路径
# with open(file_path2, 'r') as file:
#     lines2 = file.readlines()
#
# # 指定源文件夹和正确分类目标文件夹
# source_folder2 = 'C:/Users/zhl98/Desktop/VOC2023/CroppedImages'  # 替换为你的源文件夹路径
# destination_folder2 = 'C:/Users/zhl98/Desktop/VOC2023/right'  # 替换为你的目标文件夹路径
#
# # 拷贝文件到目标文件夹
# copy_images_from_list(lines, source_folder, destination_folder)
# copy_images_from_list(lines2, source_folder2, destination_folder2)
