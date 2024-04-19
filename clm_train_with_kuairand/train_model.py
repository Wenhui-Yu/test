from utils import *
import numpy as np
import random as rd

from models.model_IntEL import model_IntEL
from models.model_PRM import model_PRM
from models.model_MLP import model_MLP

def train_model(para):
    ## paths of data
    train_path = para['DIR'] + 'kuairand_ltr_data_train.json'
    test_path = para['DIR'] + 'kuairand_ltr_data_test.json'
    save_model_path = './model_ckpt/model_' + para["MODEL"] + '/model_' + para["MODEL"] + '.ckpt'

    ## Load data
    train_data, num = read_data(train_path)
    test_data, _ = read_data(test_path)
    print("len(train_data)=",len(train_data), ", num=", num)

    data = {"num": num}

    ## define the model
    if para["MODEL"] == 'IntEL': model = model_IntEL(data=data, para=para)
    if para["MODEL"] == 'PRM': model = model_PRM(data=data, para=para)
    if para["MODEL"] == 'MLP': model = model_MLP(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # saver
    # saver = tf.train.Saver(max_to_keep = 5)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # process data
    train_data_input = []
    real_len_input = []
    real_len_min = 10000
    pxtr_bucket_range = np.linspace(0, 1, num=10000)
    for sample in range(len(train_data)):
        sample_list, real_len = generate_sample_with_max_len(train_data[sample], para)      # [100, 13]
        sample_list = generate_sample_with_pxtr_bins(sample_list, para, pxtr_bucket_range)  # [100, 13+5], [pltr_index, pwtr_index, pcmtr_index, plvtr_index, plvtr_index]
        train_data_input.append(sample_list)  
        real_len_input.append(real_len)
        real_len_min = min(real_len_min, real_len)
    train_data_input = np.array(train_data_input)  # [-1, 100, 13+5]
    real_len_input = np.array(real_len_input)
    print ("len(train_data_input)=", len(train_data_input), ", len(real_len_input)=", len(real_len_input))
    print ("real_len_min=", real_len_min)

    test_data_input = []
    test_len_input = []
    pxtr_bucket_range = np.linspace(0, 1, num=10000)
    for sample in range(len(test_data)):
        sample_list, real_len = generate_sample_with_max_len(test_data[sample], para)       # [100, 13]
        sample_list = generate_sample_with_pxtr_bins(sample_list, para, pxtr_bucket_range)  # [100, 13+5], [pltr_index, pwtr_index, pcmtr_index, plvtr_index, plvtr_index]
        test_data_input.append(sample_list)
        test_len_input.append(real_len)
    test_data_raw = np.array(test_data_input)  # [-1, 100, 13+5]
    test_len_raw = np.array(test_len_input)

    ## split the training samples into batches
    batches = list(range(0, len(train_data), para['BATCH_SIZE']))
    batches.append(len(train_data))

    ## training iteratively
    for epoch in range(para['N_EPOCH']):
        ## train
        pred_list = []
        for batch_num in range(len(batches)-1):
            train_batch_data = train_data_input[batches[batch_num]: batches[batch_num+1]]  # [-1, 100, 13+5]
            real_len_batch = real_len_input[batches[batch_num]: batches[batch_num+1]] # [-1]
            # preedict first
            _, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias, pred = sess.run(
                [model.updates, model.loss, model.loss_click, model.loss_sim_order, model.loss_pxtr_reconstruct,
                 model.loss_pxtr_bias, model.pred],
                feed_dict={
                    model.intent: train_batch_data[:, :, 0],
                    model.click_label_list: train_batch_data[:, :, 2],
                    model.like_label_list: train_batch_data[:, :, 3],
                    model.follow_label_list: train_batch_data[:, :, 4],
                    model.comment_label_list: train_batch_data[:, :, 5],
                    model.forward_label_list: train_batch_data[:, :, 6],
                    model.longview_label_list: train_batch_data[:, :, 7],
                    model.real_length: real_len_batch,
                    model.keep_prob: 0.999,
                    model.like_pxtr_list: train_batch_data[:,:,13],
                    model.follow_pxtr_list: train_batch_data[:,:,14],
                    model.comment_pxtr_list: train_batch_data[:,:,15],
                    model.forward_pxtr_list: train_batch_data[:,:,16],
                    model.longview_pxtr_list: train_batch_data[:,:,17],
                    model.like_pxtr_dense_list: train_batch_data[:,:,8],
                    model.follow_pxtr_dense_list: train_batch_data[:,:,9],
                    model.comment_pxtr_dense_list: train_batch_data[:,:,10],
                    model.forward_pxtr_dense_list: train_batch_data[:,:,11],
                    model.longview_pxtr_dense_list: train_batch_data[:,:,12],
            })
            pred_list.append(pred)
        pred_list = np.concatenate(pred_list, axis=0)
        # print_click_ndcg(epoch, [10], train_data_input, pred_list, 'train')
        # print_pxtr_ndcg(epoch, train_data_input, pred_list, 'train')
        ## eval
        sampling = rd.sample(list(range(len(test_data_input))), para['TEST_USER_BATCH'])
        test_data_input = test_data_raw[sampling]
        test_len_input = test_len_raw[sampling]
        test_pred_list = []
        test_loss, test_loss_click, test_loss_sim_order, test_loss_pxtr_reconstruct, test_loss_pxtr_bias, test_pred = sess.run(
            [model.loss, model.loss_click, model.loss_sim_order, model.loss_pxtr_reconstruct, model.loss_pxtr_bias,
             model.pred],
            feed_dict={
                model.intent: test_data_input[:,:,0],
                model.click_label_list: test_data_input[:,:,2],
                model.like_label_list: test_data_input[:, :, 3],
                model.follow_label_list: test_data_input[:, :, 4],
                model.comment_label_list: test_data_input[:, :, 5],
                model.forward_label_list: test_data_input[:, :, 6],
                model.longview_label_list: test_data_input[:, :, 7],
                model.real_length: test_len_input,
                model.keep_prob: 1.0,
                model.like_pxtr_list: test_data_input[:,:,13],
                model.follow_pxtr_list: test_data_input[:,:,14],
                model.comment_pxtr_list: test_data_input[:,:,15],
                model.forward_pxtr_list: test_data_input[:,:,16],
                model.longview_pxtr_list: test_data_input[:,:,17],
                model.like_pxtr_dense_list: test_data_input[:,:,8],
                model.follow_pxtr_dense_list: test_data_input[:,:,9],
                model.comment_pxtr_dense_list: test_data_input[:,:,10],
                model.forward_pxtr_dense_list: test_data_input[:,:,11],
                model.longview_pxtr_dense_list: test_data_input[:,:,12],
        })
        test_pred_list.append(test_pred) # pred = [-1, max_len]
        test_pred_list = np.concatenate(test_pred_list, axis=0) # test_pred_list = [-1, max_len]

        # print_loss(epoch, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias)
        # print_loss(epoch, test_loss, test_loss_click, test_loss_sim_order, test_loss_pxtr_reconstruct, test_loss_pxtr_bias)
        # save_ckpt(epoch, sess, saver, save_model_path)
        # print_click_ndcg(epoch, para['TOP_K'], test_data_input, test_pred_list, 'test')
        print_pxtr_ndcg(epoch, test_data_input, test_pred_list, 'test')
        print_pxtr_auc(epoch, test_data_input, test_pred_list, 'test')
        print_pxtr_map(epoch, test_data_input, test_pred_list, 'test')

        if not loss < 10 ** 10:
            print ("ERROR, loss big, loss=", loss)
            break


