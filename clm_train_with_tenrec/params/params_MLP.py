from params.params_common import *

LR = 0.0001
pxtr_weight = [1.0, 1.0, 1.0]
exp_weight = {'click': 1.0, 'unsuper': 2.0}[LOSS]
sim_order_weight = {'click': 0.0, 'unsuper': 0.1}[LOSS]
pxtr_reconstruct_weight = {'click': 0.0, 'unsuper': 0.01}[LOSS]
mode = ['LR', 'MLP'][1]
# model save

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR,
            'PXTR_DIM': PXTR_DIM, 'DIM': DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE,
            'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'TOP_K': TOP_K, 'PXTR_LIST': PXTR_LIST, 'OPTIMIZER': 'Adam', 'DIR': DIR,
            'pxtr_weight': pxtr_weight, 'exp_weight': exp_weight, 'sim_order_weight': sim_order_weight,
            'pxtr_reconstruct_weight': pxtr_reconstruct_weight, 'mode': mode}