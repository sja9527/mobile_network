import os
import sys
from config import LFW_DATA_DIR
import numpy as np
import cv2
import scipy.io
import copy
import torch.utils.data
from core import model
from dataloader.LFW_loader import LFW
import argparse
from tqdm import tqdm
from lfw_eval import parseList
def main():
    # 加载训练好的模型
    net = model.MobileFacenet()
    ckpt = torch.load("model/best/068.ckpt", map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'])

    net.eval()

    # 　加载数据集
    nl, nr, flods, flags = parseList(LFW_DATA_DIR)  # 得到左边人和右边人
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0  # 图片总数

    # 这一步是在网络算出每个人的人脸特征，得到了左边的人脸特征和右边的特征，做余弦相似度，就能判断是不是同一个人
    for i, data in tqdm(enumerate(lfw_loader)):
        # data (4， 32, 3, 112, 96)
        count += data[0].size(0)  # batch的大小
        # print('extracing deep features from the face pair {}...'.format(count))
        res = [net(d).data.cpu().numpy() for d in data]  # d 表示一个图像
        featureL = np.concatenate((res[0], res[1]), 1)  # 纵向拼接，也就是接在列后面
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)  # 横向拼接，也就是接在行后面
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureRs.shape) # （6000， 256）这里256的原因是在加载数据集时将逆转宽度的向量也加入进来了，类似与得到了一张左右对称的图

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat("result.mat", result)

    ACCs = np.zeros(10)
    result = scipy.io.loadmat("result.mat")
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0) # 获得人脸特征的均值
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu # 减去均值
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1) # 除以根号下的平方
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)
        # 其实上面几步就是归一化，映射到-1到1之间

        scores = np.sum(np.multiply(featureLs, featureRs), 1) # 求向量内积
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000) # 从验证集中获得精度最高的阈值
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold) # 根据阈值，算在测试机中的准确率
    return ACCs

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold) # 得到TP(true positive)的数量
    n = np.sum(scores[flags == -1] < threshold) # 得到TN(true negative)的数量
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum  # 阈值在-1到1之间
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys)) # 搜索精度最高的那些阈值
    bestThreshold = np.mean(thresholds[max_index]) # 去阈值平均
    return bestThreshold


if __name__ == '__main__':
    main()

# 利用featureLs， featureRs， flods和flags计算余弦相似度








