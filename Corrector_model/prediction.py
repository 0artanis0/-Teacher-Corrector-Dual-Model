# coding=utf-8
"""
调用模型，检测错误图片，并记录，同时生成日志以供根据分类结果清理数据集

    @Project: Dataset_label_error_cleanup_classifier
    @File   : prediction.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PredictionDateSet
from resnet import resnet50
from data_process import append_to_txt, add_content_to_first_line, clear_txt_file


def make_label_txt(folder_path, txt_file_path):
    with open(txt_file_path, 'w') as txt_file:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                line = f"{filename} 0\n"
                txt_file.write(line)


def main():
    # Example usage
    base_path = 'D:/Paper with code/Trials/datasetB15/VOC2007'
    folder_path = os.path.join(base_path, 'CroppedImages')
    txt_file_path = os.path.join(folder_path, 'val.txt')
    make_label_txt(folder_path, txt_file_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 1

    # Test dataset
    test_datasets = PredictionDateSet(folder_path, txt_file_path, transforms=None)
    test_data = DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=4)

    # Model setup
    net = resnet50(num_class).to(device)
    net.load_state_dict(torch.load("model/classification C4.pth"))
    net.eval()

    clear_txt_file(os.path.join(folder_path, 'right.txt'))
    clear_txt_file(os.path.join(folder_path, 'false.txt'))
    a = 0

    for images, labels in test_data:
        # To GPU
        images = images.to(device)
        outputs = net(images)

        image_name = os.path.basename(str(labels)[:-3])
        outputs = outputs.squeeze(dim=1)
        outputs = torch.sigmoid(outputs)

        # Prediction
        pred = (outputs > 0.8).float()

        if pred == 0:
            print('其他')
            append_to_txt(os.path.join(folder_path, 'false.txt'), image_name)
            a += 1
        elif pred == 1:
            print('电线杆')
            append_to_txt(os.path.join(folder_path, 'right.txt'), image_name)

    error_rate = a / len(test_data)
    print(f"'错误率' {error_rate}")

    correct_classification = len(test_data) - a
    accuracy = correct_classification / len(test_data)

    add_content_to_first_line(os.path.join(folder_path, 'right.txt'),
                              f'There are a total of {len(test_data)} pictures, {correct_classification} '
                              f'of which are correctly classified, with an accuracy of {accuracy}')

    add_content_to_first_line(os.path.join(folder_path, 'false.txt'),
                              f'There are a total of {len(test_data)} pictures, {a} of which are incorrectly classified, '
                              f'with an error rate of {error_rate}')


if __name__ == '__main__':
    main()
