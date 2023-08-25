import json
import torch
import numpy
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

    def __get_tensor_of_data(self, path, label, convert = None, resize = 256):
        image = Image.open(path)
        image = image.convert('RGB')
        rows = image.height
        cols = image.width
        if rows < cols:
            actual_cols = (cols - rows)//2
            image = image.crop((actual_cols, 0, (actual_cols + rows), rows))
            image = image.resize((resize, resize))
        elif rows > cols:
            actual_cols = (rows - cols)//2
            image = image.crop((0, actual_cols, cols, (actual_cols + cols)))
            image = image.resize((resize, resize))
        else:
            image = image.resize((resize, resize))
        tensor_transform = torchvision.transforms.ToTensor()
        image_tensor = tensor_transform(image)
        if image_tensor.shape[0] == 1:
            print(image_tensor)
        if convert != None:
            image_tensor = convert(image_tensor)
        label_tensor = torch.zeros(100)
        label_tensor[label] = 1
        return image_tensor, label_tensor

    def __load_data_from_file(self, data_file):
        json_file_stream = open(data_file, 'r')
        dataset = json.load(json_file_stream)
        for index in range(len(dataset["image_names"])):
            if (index%600) < 100:
                absolute_path = self.__dataSet_absolute_path
                absolute_path = absolute_path + dataset["image_names"][index]
                self.__all_train_data_file_label_list.append([absolute_path, dataset["image_labels"][index]])
            if (index%600) >= 500:
                absolute_path = self.__dataSet_absolute_path
                absolute_path = absolute_path + dataset["image_names"][index]
                self.__all_test_data_file_label_list.append([absolute_path, dataset["image_labels"][index]])
    
    def __getitem__(self, index):
        path, label = self.__all_train_data_file_label_list[index]
        index_tensor = torch.tensor([index])
        lable_tensor = torch.tensor([label])
        return index_tensor, lable_tensor
    
    def get_batch_data_and_label(self, batch_index_tensor):
        batch_data = torch.tensor(batch_index_tensor.shape[0].numpy(), 3, 224, 224)
        batch_label = torch.tensor(batch_index_tensor.shape[0].numpy(), 100)
        count = 0
        for index in batch_index_tensor.numpy():
            actual_index = index//2048
            augmentation_type = index%2048
            data_tensor, label_tensor = self.get_augmentation_data(actual_index, augmentation_type)
            batch_data[count] = data_tensor
            batch_label[count] = label_tensor
        return batch_data, batch_label
    
    def get_augmentation_data(self, actual_index, augmentation_type):
        path, label = self.__all_train_data_file_label_list[actual_index]
        if augmentation_type < 1024:            
            data_tensor, label_tensor = self.__get_tensor_of_data(path, label)
        else:
            data_tensor, label_tensor = self.__get_tensor_of_data(path, label, torchvision.transforms.RandomHorizontalFlip())
        x_offset = augmentation_type%32
        y_offset = augmentation_type//32
        data_tensor = data_tensor[0:3, y_offset:(y_offset+224), x_offset:(x_offset+224)]
        return data_tensor, label_tensor
           
    def __len__(self):
        return 2048*len(self.__all_train_data_file_label_list)
    
    def get_test_data(self):
        test_data = []
        for index in range(len(self.__all_test_data_file_label_list)):
            path, label = self.__all_test_data_file_label_list[index]
            image_tensor, label_tensor = self.__get_tensor_of_data(path, label, resize=224)
            test_data.append([image_tensor, label_tensor])
        return test_data
    
    def get_train_data(self, sum):
        train_data = []
        for count in range(sum):
            index = random.randint(0, len(self.__all_train_data_file_label_list) - 1)
            path, label = self.__all_train_data_file_label_list[index]
            image_tensor, label_tensor = self.__get_tensor_of_data(path, label, resize=224)
            train_data.append([image_tensor, label_tensor])
        return train_data