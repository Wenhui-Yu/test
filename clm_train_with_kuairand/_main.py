from params.params_common import MODEL
if MODEL == "IntEL": from params.params_IntEL import all_para
if MODEL == "PRM": from params.params_PRM import all_para
if MODEL == "MLP": from params.params_MLP import all_para

# from params import all_para
from train_model import *
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    ## print model hyperparameters
    print_params(all_para)
    ## train the model
    train_model(all_para)

