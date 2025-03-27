# coding=utf-8
"""
根据标注，将指定样本从数据集图片中裁下，储存在目标文件夹下CroppedImages目录

    @Project: Dataset_label_error_cleanup_classifier
    @File   : ImageCrop.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image


class ImageCrop(object):
    """用于按照XML标注内容将pole文件全部裁剪到数据集内的CroppedImages文件夹下。

    数据集内文件夹以及标注格式应该为PASCAL VOC"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'JPEGImages')
        self.xml_path = os.path.join(dataset_path, 'Annotations')
        self.output_path = os.path.join(dataset_path, 'CroppedImages')
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
                print(f"目录 '{self.output_path}' 创建成功。")
            except OSError as e:
                print(f"创建目录 '{self.output_path}' 时出错: {e}")
        else:
            print(f"目录 '{self.output_path}' 已存在。")

    def crop_images_in_folder(self):
        image_folder = self.img_path
        xml_folder = self.xml_path
        output_folder = self.output_path

        # 遍历文件夹内的所有图片
        for image_filename in os.listdir(image_folder):
            if image_filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_filename)

                # 查找相应的XML文件
                xml_filename = os.path.splitext(image_filename)[0] + '.xml'
                xml_path = os.path.join(xml_folder, xml_filename)

                # 如果找到XML文件，则进行裁剪
                if os.path.exists(xml_path):
                    self.crop_image(image_path, xml_path, output_folder)

    def crop_image(self, image_path, xml_path, output_folder):
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片尺寸
        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)

        # 逐个处理pole元素
        i = 1
        for obj in root.findall(".//object[name='pole']"):
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # 裁剪图片
            image = Image.open(image_path)
            cropped_image = image.crop((xmin, ymin, xmax, ymax))

            # 保存裁剪后的图片
            output_filename = f"{output_folder}/{i}%{os.path.basename(image_path)}"
            cropped_image.save(output_filename)
            print(f"保存裁剪后的图片: {output_filename}")
            i += 1


if __name__ == '__main__':
    dataset_path = 'D:/Paper with code/Trials/datasetB15/VOC2007'
    imgcrop = ImageCrop(dataset_path)
    imgcrop.crop_images_in_folder()

    dataset_path2 = 'D:/Paper with code/Trials/datasetB15/VOC2012'
    imgcrop = ImageCrop(dataset_path2)
    imgcrop.crop_images_in_folder()
