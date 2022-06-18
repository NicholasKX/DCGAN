# -*- coding: utf-8 -*-
"""
Created on 2022/5/10 16:05 
@Author: Wu Kaixuan
@File  : dataloader.py 
@Desc  : dataloader 
"""
from torch.utils.data import Dataset
import os
from PIL import Image

class LEVIR(Dataset):

    def __init__(self,root,img_size=512,transform=None):
        self.root = root
        self.img_size = img_size
        self.data = os.listdir(self.root)
        self.length = len(self.data)
        self.images_fps = [os.path.join(self.root, image_id) for image_id in self.data]
        self.transform = transform


    def __getitem__(self, item):
        image = Image.open(self.images_fps[item]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.length

if __name__ == '__main__':
    from torchvision.transforms import transforms
    data = LEVIR(root='../datasets/train/B',
                 transform=transforms.Compose([
                     transforms.Resize(512),
                     # transforms.CenterCrop(img_size),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ]))
    print(data[0].shape)