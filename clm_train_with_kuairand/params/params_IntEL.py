from params.params_common import *

LR = 0.01
exp_weight = {'primary': 0.0, 'click': 1.0, 'multi-obj': 0.0, 'unsuper': 1.0}[LOSS]
sim_order_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 2.0}[LOSS]
pxtr_reconstruct_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 0.1}[LOSS]
primary_weight = {'primary': 1.0, 'click': 0.0, 'multi-obj': 0.0, 'unsuper': 0.0}[LOSS]
multi_object_weight = {'primary': 0.0, 'click': 0.0, 'multi-obj': 1.0, 'unsuper': 0.0}[LOSS]
pxtr_weight_for_ranking_sim_loss = [1.0, 1.0, 1.0, 1.0, 1.0]
pxtr_weight_for_multi_object = [1.0, 1.0, 1.0, 1.0, 1.0]
bias_weight = 10.0
layer_num = 5

# model save
all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'CANDIDATE_ITEM_LIST_LENGTH': CANDIDATE_ITEM_LIST_LENGTH, 'LR': LR,
            'PXTR_DIM': PXTR_DIM, 'DIM': DIM, 'PXTR_BINS': PXTR_BINS, 'BATCH_SIZE': BATCH_SIZE, 'PRED_BATCH_SIZE': PRED_BATCH_SIZE,
            'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'TOP_K': TOP_K, 'PXTR_LIST': PXTR_LIST, 'OPTIMIZER': 'Adam', 'DIR': DIR,
            'pxtr_weight': pxtr_weight_for_ranking_sim_loss, 'exp_weight': exp_weight, 'sim_order_weight': sim_order_weight,
            'pxtr_reconstruct_weight': pxtr_reconstruct_weight, 'bias_weight': bias_weight, 'layer_num': layer_num,
            'primary_weight': primary_weight, 'multi_object_weight': multi_object_weight,  'pxtr_prompt': pxtr_weight_for_multi_object}
