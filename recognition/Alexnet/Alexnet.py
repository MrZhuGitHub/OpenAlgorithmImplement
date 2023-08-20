import json
import dataset_load
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils import data
# import cv2
import torch
from PIL import Image

# def show_image(image):
#         arrayImage = image.numpy()
#         arrayImage = np.transpose(arrayImage, (1, 2, 0))
#         cv2.imshow('image', arrayImage)
        # plt.imshow(arrayImage)


def init_weight(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')

if __name__=="__main__":
    net = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5),
        torch.nn.ReLU(),
        # torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
        torch.nn.ReLU(),
        # torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=9216, out_features=4096),
        torch.nn.Dropout1d(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=4096, out_features=1024),
        torch.nn.Dropout1d(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1024, out_features=100),
        torch.nn.ReLU(),
    )
    net.cuda()
    net.apply(init_weight)

    crossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="mean")
    crossEntropyLoss.cuda()

    trainer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    miniImageNet_dataSet = dataset_load.miniImageNet_load()
    train_iter = data.DataLoader(dataset=miniImageNet_dataSet, batch_size=100, num_workers=8, shuffle=True)

    test_data_list = miniImageNet_dataSet.get_test_data()

    num_epoch = 500
    for epoch in range(num_epoch):
        net.train()
        for index, batch_data_and_label in enumerate(train_iter):
            batch_data, label = batch_data_and_label
            batch_data = batch_data.cuda()
            label = label.cuda()
            loss = crossEntropyLoss(net(batch_data), label)
            trainer.zero_grad()
            loss.backward()
            trainer.step()

        net.eval()
        with torch.no_grad():
            train_true_count = 0
            train_data_sample_list = miniImageNet_dataSet.get_train_data(1000)
            for index, value in enumerate(train_data_sample_list):
                sample_data, label = value
                sample_data = sample_data.cuda()
                result = torch.argmax(net(torch.unsqueeze(sample_data, 0))).cpu()
                type = result.numpy()
                if 1 == label[type].numpy():
                    train_true_count = train_true_count + 1          

            test_true_count = 0
            for index, value in enumerate(test_data_list):
                sample_data, label = value
                sample_data = sample_data.cuda()
                result = torch.argmax(net(torch.unsqueeze(sample_data, 0))).cpu()
                type = result.numpy()
                if 1 == label[type].numpy():
                    test_true_count = test_true_count + 1
            print('\n epoch = %d, train = %f, test = %f' % (epoch, (train_true_count/len(train_data_sample_list)), (test_true_count/len(test_data_list))))



