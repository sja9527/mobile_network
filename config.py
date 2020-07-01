import torch

BATCH_SIZE = 256
SAVE_FREQ = 1
TEST_FREQ = 5
TOTAL_EPOCH = 60

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

RESUME = ''
SAVE_DIR = './model' # 保存模型的目录
MODEL_PRE = 'CASIA_B512_'


CASIA_DATA_DIR = 'CASIA'
LFW_DATA_DIR = 'LFW'

GPU = 2

