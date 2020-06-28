import numpy as np
# import scipy.misc
import os
import torch
import sys
sys.path.append('..')
from config import CASIA_DATA_DIR
import imageio

from torchvision import transforms
from matplotlib import pyplot as plt

class CASIA_Face(object):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'CASIA-WebFace-112X96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))



    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        # img = scipy.misc.imread(img_path)
        img = imageio.imread(img_path)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    # data_dir = '/home/brl/USER/fzc/dataset/CASIA'
    dataset = CASIA_Face(root=CASIA_DATA_DIR)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))

    sample = next(iter(trainloader))
    print(sample[0].shape, sample[1].shape)

    # to_img = transforms.ToPILImage()
    # img = to_img(sample[0][0])# C H W .numpy() 
    # img.show()

    # img = np.transpose(sample[0][0].numpy(), (1, 2, 0))
    # plt.imshow(img)
    # plt.show()

    
    # print(trainloader[0].shape, trainloader[1].shape)
    # for data in trainloader:
    #     print(data[0].shape)
