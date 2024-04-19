* Environment:
  Python 3.8.5
* Libraries:
  tensorflow 1.15.5
  numpy 1.18.0
  pandas 1.2.0

There are two steps to conduct the experiment: (1) pretraining to get pxtr data (can be skipped), and (2) training ranking ensemble models.

* Pretraining:
    You can download the pxtr data directly or generate it:
    1. If you chose downloading the pxtr data:
        a. Download from from https://drive.google.com/file/d/1tW9F9ovWRzNO71bplcJg_T_fsyopzAcI/view?usp=drive_link
        b. Unzip the .zip file, and put the datasets in clm_pretraining_with_${dataset}/dataset/${dataset}
    2. If you chose generating the pxtr data
        a. Download raw data from https://kuairand.com/ and https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
        b. For dataset Tenrec, run tenrec_data_extraction.ipynb for pre-processing.
        c. Run ${dataset}_data_processor.ipynb to generate data for mmoe model.
        d. Run _main.py to train the model and save ckpt
        e. Run prediction_data.py to predict xtr scores
        f. ${dataset}_make_ltr_data.ipynb to generate data for ranking ensemble model.
* Training:
    1. Under path clm_train_with_${dataset}
    2. Set hyper-parameters in params/params_common.py and params/params_${model}.py
        a. UREM_PRM: MODEL = ['IntEL', 'PRM', 'MLP'][1]
        b. UREM_IntEL: default
        c. PRM: MODEL = ['IntEL', 'PRM', 'MLP'][1] & LOSS = ['primary', 'click', 'multi-obj', 'unsuper'][1]
        d. IntEL: LOSS = ['primary', 'click', 'multi-obj', 'unsuper'][1]
        e. MLP: MODEL = ['IntEL', 'PRM', 'MLP'][2] & LOSS = ['primary', 'click', 'multi-obj', 'unsuper'][1]
        f. LR: MODEL = ['IntEL', 'PRM', 'MLP'][2] & LOSS = ['primary', 'click', 'multi-obj', 'unsuper'][1] & mode = ['LR', 'MLP'][0]
    3. Run _main.py

* ${dataset} indicate "KuaiRand"/"kuaiRand"/"kuai" and "Tenrec"/"tenrec".