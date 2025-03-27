# -*-coding:utf-8-*-
"""
一些txt及xml文件处理函数

    @Project: Dataset_label_error_cleanup_classifier
    @File   : data_process.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import xml.etree.ElementTree as ET
import os


def clear_txt_file(file_path):
    """清空文本文件的内容。

    Args:
        file_path (str): 文件路径。
    """
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')
        print(f"文件 '{file_path}' 已清空")
    else:
        print(f"文件 '{file_path}' 不存在")


def modify_file_extension(input_string):
    """修改文件扩展名为.xml。

    Args:
        input_string (str): 输入字符串。

    Returns:
        str: 修改后的字符串。
    """
    if '.' in input_string:
        last_dot_index = input_string.rfind('.')
        modified_string = input_string[:last_dot_index] + '.xml'
        return modified_string
    else:
        return input_string + '.xml'


def append_to_txt(file_path, content):
    """将内容追加到文本文件末尾。

    Args:
        file_path (str): 文件路径。
        content (str): 要追加的内容。
    """
    with open(file_path, 'a+') as file:
        file.seek(0, 2)
        if file.tell() > 0:
            file.write('\n')
        file.write(content)


def add_content_to_first_line(file_path, content_to_add):
    """在文本文件的第一行添加内容。

    Args:
        file_path (str): 文件路径。
        content_to_add (str): 要添加的内容。
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as new_file:
            new_file.write(content_to_add + '\n')
        return

    with open(file_path, 'r') as file:
        original_content = file.read()

    with open(file_path, 'w') as file:
        file.write(content_to_add + '\n')
        file.write(original_content)


def sum_numbers_class_in_txt(file_path, keywords):
    """计算文本文件中指定关键字前数字的总和。

    Args:
        file_path (str): 文件路径。
        keywords (str or list): 要搜索的关键字或关键字列表。

    Returns:
        dict: 包含每个关键字总和的字典。
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    total_sum = {keyword: 0 for keyword in keywords}

    with open(file_path, 'r') as file:
        for line in file:
            for keyword in keywords:
                keyword_index = line.find(keyword)
                if keyword_index != -1:
                    numbers_before_keyword = line[:keyword_index].split()[-1]
                    try:
                        number = int(numbers_before_keyword)
                        total_sum[keyword] += number
                    except ValueError:
                        print(f"将 '{numbers_before_keyword}' 转换为整数时出错。")

    return total_sum


def add_object_to_xml(xml_file_path, object_name, xmin, ymin, xmax, ymax):
    """在XML文件中添加<object>元素。

    Args:
        xml_file_path (str): XML文件路径。
        object_name (str): 添加的对象名称。
        xmin (int): Bounding box的xmin。
        ymin (int): Bounding box的ymin。
        xmax (int): Bounding box的xmax。
        ymax (int): Bounding box的ymax。
    """
    if not os.path.exists(xml_file_path):
        root = ET.Element("annotation")
        tree = ET.ElementTree(root)
        tree.write(xml_file_path)

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    existing_objects = root.findall(".//object[name='" + object_name + "']")
    existing_objects_with_same_bndbox = [
        obj for obj in existing_objects
        if obj.findtext("bndbox/xmin") == str(xmin)
           and obj.findtext("bndbox/ymin") == str(ymin)
           and obj.findtext("bndbox/xmax") == str(xmax)
           and obj.findtext("bndbox/ymax") == str(ymax)
    ]

    if existing_objects_with_same_bndbox:
        print("相同内容的对象已存在。跳过添加。")
        return

    new_object = ET.Element("object")
    name = ET.SubElement(new_object, "name")
    name.text = object_name
    pose = ET.SubElement(new_object, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(new_object, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(new_object, "difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(new_object, "bndbox")
    xmin_element = ET.SubElement(bndbox, "xmin")
    xmin_element.text = str(xmin)
    ymin_element = ET.SubElement(bndbox, "ymin")
    ymin_element.text = str(ymin)
    xmax_element = ET.SubElement(bndbox, "xmax")
    xmax_element.text = str(xmax)
    ymax_element = ET.SubElement(bndbox, "ymax")
    ymax_element.text = str(ymax)

    root.append(new_object)
    tree.write(xml_file_path)

# 测试
# xml_file_path = "2008_003655.xml"
# add_object_to_xml(xml_file_path, "pole", 326, 0, 369, 332)
# add_object_to_xml(xml_file_path, "pole", 82, 0, 124, 332)
# print(modify_file_extension('002534.jpg'))
