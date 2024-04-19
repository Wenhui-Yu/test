
GPU_INDEX = "0"
dataset = 1         # 0:Amazon, 1:KuaiRand
DATASET = ['Tamll', 'KuaiRand'][dataset]
MODEL = 'MMOE'
LR = [0.001, 0.01][dataset]
LAMDA = [0.2, 0.1][dataset]
ACTION_LIST_MAX_LEN = 150
EMB_DIM = 64
BATCH_SIZE = 10000
validate_test = 0   # 0:Validate, 1: Test
TEST_USER_BATCH = [4096, 4096][dataset]
SAMPLE_RATE = 1
N_EPOCH = 50
BEST_EPOCH = [66, 48][dataset]
LOSS_WEIGHT = [[2, 1, 2, 2, 1],
               [1.7, 2.2, 2.2, 1.4, 0.9]][dataset]
TOP_K = [10, 20, 50, 100]
DIR = './dataset/'+DATASET+'/'

# model save

#pred
PRED_BATCH_SIZE = 10000


all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'ACTION_LIST_MAX_LEN': ACTION_LIST_MAX_LEN, 
            'EMB_DIM': EMB_DIM, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE,'LOSS_FUNCTION': 'CrossEntropy',
            'OPTIMIZER': 'Adam', 'SAMPLER': 'MMOE', 'AUX_LOSS_WEIGHT': 0, 'BEST_EPOCH': BEST_EPOCH, 'LOSS_WEIGHT': LOSS_WEIGHT}