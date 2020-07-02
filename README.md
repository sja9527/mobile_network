# MobileFaceNet

## 运行环境

* Python 3.8
* pytorch 1.3+
* GPU or CPU

## 使用方法

### 1、下载数据集

[Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg) and [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).

### 2、训练（这一步可以没有，在result中提供了训练好的）

改变**CAISIA_DATA_DIR** and **LFW_DATA_DAR** (在`config.py`文件中)
  
运行指令训练

```
python train_my.py
```
      
### 3、测试

在LFW数据集上测试
    
      
```
sh ./run.sh
```

### 4、运行结果

```
1    98.67
2    98.33
3    98.50
4    98.50
5    98.33
6    98.83
7    98.83
8    98.67
9    99.50
10    98.83
--------
AVE    98.70

```

10折交叉验证的平均值为98.7，论文原文中的batch_size为512，然而因显存不够我调到了256，其他设置均和原文相同

## 参考资料

  * [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
  * [SphereFace](https://github.com/wy1iu/sphereface)
  * [Insightface](https://github.com/deepinsight/insightface)
  * [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
  
  
  

