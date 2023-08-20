import json
import torch
import numpy
# import cv2
import torchvision
import torch.utils.data as data
import random
from PIL import Image


class miniImageNet_load(data.Dataset):
    def __init__(self):
        self.__all_train_data_file_label_list = []
        self.__all_test_data_file_label_list = []
        self.__all_train_data_tensor_label_list = []
        self.__all_test_data_tensor_label_list = []
        self.__dataSet_absolute_path = "/root/"
        self.__train_data_json_file = "/root/filelists/miniImagenet/base.json"
        self.__test_data_json_file = "/root/filelists/miniImagenet/novel.json"
        self.__val_data_json_file = "/root/filelists/miniImagenet/val.json"
        self.__load_data_from_file(self.__train_data_json_file)
        self.__load_data_from_file(self.__test_data_json_file)
        self.__load_data_from_file(self.__val_data_json_file)

    def __get_tensor_of_data(self, path, label, convert = None):
        image = Image.open(path)
        image = image.convert('RGB')
        rows = image.height
        cols = image.width
        if rows < cols:
            actual_cols = (cols - rows)//2
            image = image.crop((actual_cols, 0, (actual_cols + rows), rows))
            image = image.resize((224, 224))
        elif rows > cols:
            actual_cols = (rows - cols)//2
            image = image.crop((0, actual_cols, cols, (actual_cols + cols)))
            image = image.resize((224, 224))
        else:
            image = image.resize((224, 224))
        tensor_transform = torchvision.transforms.ToTensor()
        image_tensor = tensor_transform(image)
        if image_tensor.shape[0] == 1:
            print(image_tensor)
        if convert != None:
            image_tensor = convert(image_tensor)
        label_tensor = torch.zeros(100)
        label_tensor[label] = 1
        return image_tensor, label_tensor       
        # image = cv2.imread(path)
        # rows = image.shape[0]
        # cols = image.shape[1]
        # if rows < cols:
        #     actual_cols = (cols - rows)//2
        #     image = image[0:rows, actual_cols:(actual_cols + rows)]
        #     image = cv2.resize(image, (224, 224))
        # elif rows > cols:
        #     actual_cols = (rows - cols)//2
        #     image = image[actual_cols:(actual_cols + cols), 0:cols]
        #     image = cv2.resize(image, (224, 224))
        # else:
        #     image = cv2.resize(image, (224, 224))
        # tensor_transform = torchvision.transforms.ToTensor()
        # image_tensor = tensor_transform(image)
        # if convert != None:
        #     image_tensor = convert(image_tensor)
        # label_tensor = torch.zeros(100)
        # label_tensor[label] = 1
        # return image_tensor, label_tensor

    def __load_data_from_file(self, data_file):
        json_file_stream = open(data_file, 'r')
        dataset = json.load(json_file_stream)
        for index in range(len(dataset["image_names"])):
            if (index%600) < 500:
                absolute_path = self.__dataSet_absolute_path
                absolute_path = absolute_path + dataset["image_names"][index]
                self.__all_train_data_file_label_list.append([absolute_path, dataset["image_labels"][index]])
            else:
                absolute_path = self.__dataSet_absolute_path
                absolute_path = absolute_path + dataset["image_names"][index]
                self.__all_test_data_file_label_list.append([absolute_path, dataset["image_labels"][index]])
                image_tensor, label_tensor = self.__get_tensor_of_data(absolute_path, dataset["image_labels"][index])
                self.__all_test_data_tensor_label_list.append([image_tensor, label_tensor])
    
    def __getitem__(self, index):
        path, label = self.__all_train_data_file_label_list[index]
        # convert_index = index % 5
        # if 1 == convert_index:
        #     return self.__get_tensor_of_data(path, label, torchvision.transforms.RandomHorizontalFlip())
        # elif 2 == convert_index:
        #     return self.__get_tensor_of_data(path, label, torchvision.transforms.RandomVerticalFlip())
        # elif 3 == convert_index:   
        #     return self.__get_tensor_of_data(path, label, torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.6, 1), ratio=(1, 1)))
        # elif 4 == convert_index:
        #     return self.__get_tensor_of_data(path, label, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        # else:
        return self.__get_tensor_of_data(path, label)
    
    def __len__(self):
        return len(self.__all_train_data_file_label_list)
    
    def get_test_data(self):
        return self.__all_test_data_tensor_label_list
    
    def get_train_data(self, sum):
        test_data = []
        for count in range(sum):
            index = random.randint(0, len(self.__all_train_data_file_label_list) - 1)
            image_tensor, label_tensor = self.__getitem__(index)
            test_data.append([image_tensor, label_tensor])
        return test_data