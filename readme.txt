* Environment:
  Python 3.8.5
* Libraries:
  tensorflow 1.15.5
  numpy 1.18.0
  pandas 1.2.0

There are two steps to conduct the experiment: (1) pretraining to get pxtr data, and (2) training ranking ensemble models.

Step 1 is to train mmoe model to predict xtr scores and step 2 is to train ranking ensemble models.

The path of pretraining step is clm_pretraining_with_${dataset} and of training step is clm_training_with_${dataset}, where ${dataset} indicate "KuaiRand"/"kuaiRand"/"kuai" and "Tenrec"/"tenrec".

We provide download of pxtr ranking data to skip step 1.


* Preparation：
    1. Download ${dataset}.zip, unzip, and put in clm_pretraining_with_${dataset}/dataset/${dataset}
* Pretraining (can be skipped):
    1. download raw data from https://kuairand.com/ and https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
    2. For dataset Tenrec, run tenrec_data_extraction.ipynb for pre-processing.
    3. Run ${dataset}_data_processor.ipynb to generate data for mmoe model.
    4. Run _main.py to train the model and save ckpt
    5. Run prediction_data.py to predict xtr scores
    6. ${dataset}_make_ltr_data.ipynb to generate data for ranking ensemble model.
* Training (for experiments in paper):
    1. Set hyper-parameters in params/params_${model}.py
    2. Run _main.py