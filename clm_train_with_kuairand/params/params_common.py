
GPU_INDEX = "0"
MODEL = ['IntEL', 'PRM', 'MLP'][0]
LOSS = ['primary', 'click', 'multi-obj', 'unsuper'][3]
DATASET = 'KuaiRand'
CANDIDATE_ITEM_LIST_LENGTH = 100
PXTR_DIM = 16
DIM = 64
PXTR_BINS = 10000
PXTR_LIST = ['pltr', 'pwtr', 'pcmtr', 'pftr', 'plvtr']

N_EPOCH = 50
BATCH_SIZE = 10000
PRED_BATCH_SIZE = 2000
TEST_USER_BATCH = 1000
TOP_K = [10, 20, 30, 40, 50]
DIR = '../clm_pretraining_with_kuairand/dataset/'+DATASET+'/'
