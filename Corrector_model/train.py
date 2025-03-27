# -*-coding:utf-8-*-
"""
Train a resnet model on a custom dataset.

    @Project: Dataset_label_error_cleanup_classifier
    @File   : train.py
    @Author : Hongli Zhao
    @E-mail : zhaohongli8711@outlook.com
    @Date   : 2023-12-22 10:04:28
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset import CarDateSet
from resnet import resnet50

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load datasets
    train_datasets = CarDateSet('./data/train', './data/train.txt', transforms=None)
    test_datasets = CarDateSet('./data/test', './data/test.txt', transforms=None)

    # Create data loaders
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=args.batch_size, shuffle=True)

    print("Train numbers:{:d}".format(len(train_datasets)))

    # Create model
    model = resnet50(num_classes=1)
    print(model)

    # Move model to the specified device
    model = model.to(device)

    # Define loss function and optimizer
    cost = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        index = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(dim=1)
            outputs = torch.sigmoid(outputs)

            loss = cost(outputs, labels.float())

            if index % 10 == 0:
                print(loss)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1

        if epoch % 1 == 0:
            end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end - start) * 2))

            model.eval()

            correct_prediction = 0.
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                outputs = outputs.squeeze(dim=1)
                outputs = torch.sigmoid(outputs)

                # Prediction
                predicted = (outputs > 0.5).float()

                # Calculate accuracy
                total += labels.size(0)
                correct_prediction += (predicted == labels).sum().item()

            print("Acc: %.4f" % (correct_prediction / total))

        # Save the model checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))

    print("Model save to %s." % (os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model_name", default='arrow_hub_other_', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--pretrained_model", default='./model/resnet50.pth', type=str)
    args = parser.parse_args()

    main(args)
