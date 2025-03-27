# coding=utf-8
"""
根据模型分类结果，清理数据集中的的错误标注，并且在清理标注后，清除不含目标样本的图片

    @Project: Dataset_label_error_cleanup_classifier
    @File   : FalseMarkClear.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import xml.etree.ElementTree as ET
import re
from data_process import clear_txt_file


class FalseMarkClear:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.xml_folder_path = os.path.join(dataset_path, 'Annotations')
        self.txt_file_path = os.path.join(dataset_path, 'CroppedImages/false_clear.txt')
        self.input_txt_path = os.path.join(dataset_path, 'CroppedImages/false.txt')
        self.output_txt_path = os.path.join(dataset_path, 'CroppedImages/false_clear.txt')
        self.jpg_folder_path = os.path.join(dataset_path, 'JPEGImages')

    def sort_txt(self):
        """对txt排序，从大到小，便于删除时从大到小删除"""
        input_file = self.input_txt_path
        output_file = self.output_txt_path

        with open(input_file, 'r') as file:
            lines = file.readlines()[1:]  # 跳过第一行，获取从第二行到最后一行的内容

        # 对每一行按行开头的数字从大到小排序
        sorted_lines = sorted(lines, key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)

        # 将结果写入新的txt文件
        with open(output_file, 'w') as file:
            # 将第一行写回文件
            file.write(
                "总共有{}张图片，其中{}张被错误分类，错误率为{}。\n".format(
                    len(sorted_lines), 0, 0))

            # 添加换行符，确保每一行都独占一行
            for line in sorted_lines:
                file.write(line.rstrip('\n') + '\n')

    def remove_false_elements(self):
        """删除被记录在目录内的pole元素"""
        xml_folder = self.xml_folder_path
        txt_file = self.txt_file_path
        self.sort_txt()

        with open(txt_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('1%') or line.startswith('2%') or line.startswith('3%') or line.startswith(
                    '4%') or line.startswith('5%'):
                parts = line.split('%')
                jpg_name = parts[1].strip()
                xml_name = jpg_name.rstrip('.jpg') + '.xml'

                index_to_remove = int(parts[0]) - 1  # 转换为从零开始的索引

                xml_path = os.path.join(xml_folder, xml_name)

                if os.path.exists(xml_path):
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # 找到并删除对应位置的 'pole' 元素
                    pole_elements = root.findall(".//object[name='pole']")
                    if index_to_remove < len(pole_elements):
                        root.remove(pole_elements[index_to_remove])

                        # 保存修改后的XML文件
                        tree.write(xml_path)
                        print(f"文件 '{xml_path}' 中的第 {index_to_remove + 1} 个 'pole' 元素已删除")
                    else:
                        print(f"文件 '{xml_path}' 中的 'pole' 元素数量不足 {index_to_remove + 1} 个")
                else:
                    print(f"文件 '{xml_path}' 不存在")

    def remove_files_without_sample(self):
        """清理没有样本的图片和标注文件,并重新生成train.txt"""
        xml_folder = self.xml_folder_path
        jpg_folder = self.jpg_folder_path
        for xml_file in os.listdir(xml_folder):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(xml_folder, xml_file)
                jpg_file = os.path.splitext(xml_file)[0] + '.jpg'
                jpg_path = os.path.join(jpg_folder, jpg_file)

                if self.has_pole_element(xml_path):
                    print(f"XML文件 '{xml_file}' 包含 'pole' 元素。")
                else:
                    print(f"删除XML文件 '{xml_file}' 和对应的JPG文件。")
                    os.remove(xml_path)
                    if os.path.exists(jpg_path):
                        os.remove(jpg_path)
        self.list_images_to_txt()

    @staticmethod
    def has_pole_element(xml_path):
        """检查是否有对应样本"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 检查XML文件中是否存在 'pole' 元素
        return any(obj.find('name').text == 'pole' for obj in root.findall('.//object'))

    def list_images_to_txt(self):
        """重新生成imageset里的train.txt"""
        folder_path = self.jpg_folder_path
        txt_file_path = os.path.join(self.dataset_path, 'ImageSets/Main/train.txt')
        clear_txt_file(txt_file_path)
        with open(txt_file_path, 'w') as txt_file:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_name = os.path.splitext(file_name)[0]
                    txt_file.write(file_name + '\n')


if __name__ == '__main__':
    dataset_path1 = 'D:/Paper with code/Trials/datasetB15/VOC2007'
    false_mark1 = FalseMarkClear(dataset_path1)
    false_mark1.remove_false_elements()
    false_mark1.remove_files_without_sample()

    dataset_path2 = 'D:/Paper with code/Trials/datasetB15/VOC2012'
    false_mark2 = FalseMarkClear(dataset_path2)
    false_mark2.remove_false_elements()
    false_mark2.remove_files_without_sample()
