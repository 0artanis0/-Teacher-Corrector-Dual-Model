# -*-coding:utf-8-*-
"""
构建模型训练的数据集

    @Project: Dataset_label_error_cleanup_classifier
    @File   : create_labels_files.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import os.path


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]
            if curr_file == 'class0':
                labels = 0
            if curr_file == 'class1':
                labels = 1
            files_list.append([os.path.join(curr_file, filename), labels])
    return files_list


if __name__ == '__main__':
    train_dir = './data/train'
    train_txt = './data/train.txt'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    val_dir = './data/test'
    val_txt = './data/test.txt'
    val_data = get_files_list(val_dir)
    write_txt(val_data, val_txt, mode='w')

    test_dir = './data/val'
    test_txt = './data/val.txt'
    test_data = get_files_list(test_dir)
    write_txt(test_data, test_txt, mode='w')
