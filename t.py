# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: test.py
# @time: 2020/11/17 15:02
# @Software: PyCharm

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models
import torch

from datapre import VOCSegDataset, crop, classes
from mob_unet import UNet

height = 224
width = 224

voc_test = VOCSegDataset(False, height, width)


valid_data = DataLoader(voc_test, batch_size=8)

PATH = "./model/unet4.pth"
# 各种标签所对应的颜色
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
cm = np.array(COLORMAP).astype('uint8')


def predict(img1, label):
    img1 = Variable(img1.unsqueeze(0)).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(3,5).to(device)

    net.load_state_dict(torch.load(PATH))
    out = net(img1)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred]

    pred = Image.fromarray(pred)
    label1 = cm[label.numpy()]
    return pred, label1


SIZE = 224
NUM_IMG = 20
# _, figs = plt.subplots(NUM_IMG, 3, figsize=(12, 22))
for i in range(51):
    img_data, img_label = voc_test[i]
    pred, label = predict(img_data, img_label)
    img_data = Image.open(voc_test.data_list[i])
    img_label = Image.open(voc_test.label_list[i])
    img_data, img_label = crop(img_data, img_label, SIZE, SIZE)
    pred.save("./pred/"+str(i)+"_pred.png",'PNG')
    img_data.save("./pred/"+str(i)+"_img.png",'PNG')
    print("the picture {} has predicted.".format(i))

print("saving predict results finish.")
