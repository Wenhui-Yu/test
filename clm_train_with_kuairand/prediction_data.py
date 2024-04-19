from params.params_common import MODEL
if MODEL == "IntEL": from params.params_IntEL import all_para
if MODEL == "PRM": from params.params_PRM import all_para
if MODEL == "MLP": from params.params_MLP import all_para

from utils import *

import json
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']

def prediction_data(para):
    print ("Model Name = ", para["MODEL"] )
    pred_data_path = para['DIR'] + 'kuairand_ltr_data_test.json'
    # ltr_data_path = DIR + 'kuairand_ltr_data.json'
    
    #load model path
    model_path = './model_ckpt/model_' + para["MODEL"] + '/model_' + para["MODEL"] + '.ckpt-5.meta'
    restore_path = './model_ckpt/model_' + para["MODEL"] + '/model_' + para["MODEL"] + '.ckpt-5'

    ## Load data
    pred_data, _ = read_data(pred_data_path)
    print ("pred_data[0:1]=", pred_data[:1])

    # process data
    pred_data_input = []
    real_len_input = []
    real_len_min = 10000
    pxtr_bucket_range = np.linspace(0, 1, num=10000)
    for sample in range(len(pred_data)):
        sample_list, real_len = generate_sample_with_max_len(pred_data[sample], para)  # [100, 13]
        sample_list = generate_sample_with_pxtr_bins(pred_data[sample], para, pxtr_bucket_range)  # [100, 13+5], [pltr_index, pwtr_index, pcmtr_index, plvtr_index, plvtr_index]
        pred_data_input.append(sample_list)  
        real_len_input.append(real_len)
        real_len_min = min(real_len_min, real_len)
    pred_data_input = np.array(pred_data_input)  # [-1, 100, 13+5]
    real_len_input = np.array(real_len_input)
    print ("len(pred_data_input)=", len(pred_data_input), ", len(real_len_input)=", len(real_len_input))
    print ("real_len_min=", real_len_min)

    ## split the pred-samples into batches
    batches = list(range(0, len(pred_data), para['PRED_BATCH_SIZE']))
    batches.append(len(pred_data))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, restore_path)

        # feed_dict
        intent = sess.graph.get_tensor_by_name('intent:0') # [-1, max_len]
        #   label
        click_label_list = sess.graph.get_tensor_by_name('click_label_list:0')
        real_length = sess.graph.get_tensor_by_name('real_length:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        #   pxtr emb feature
        like_pxtr_list = sess.graph.get_tensor_by_name('like_pxtr_list:0') # bin
        follow_pxtr_list = sess.graph.get_tensor_by_name('follow_pxtr_list:0')
        comment_pxtr_list = sess.graph.get_tensor_by_name('comment_pxtr_list:0')
        forward_pxtr_list = sess.graph.get_tensor_by_name('forward_pxtr_list:0')
        longview_pxtr_list = sess.graph.get_tensor_by_name('longview_pxtr_list:0')
        #   pxtr dense feature
        like_pxtr_dense_list = sess.graph.get_tensor_by_name('like_pxtr_dense_list:0')
        follow_pxtr_dense_list = sess.graph.get_tensor_by_name('follow_pxtr_dense_list:0')
        comment_pxtr_dense_list = sess.graph.get_tensor_by_name('comment_pxtr_dense_list:0')
        forward_pxtr_dense_list = sess.graph.get_tensor_by_name('forward_pxtr_dense_list:0')
        longview_pxtr_dense_list = sess.graph.get_tensor_by_name('longview_pxtr_dense_list:0')

        # 2 pred
        pred = sess.graph.get_tensor_by_name('sab1/Sigmoid:0')

        # for
        pred_list = []
        for batch_num in range(len(batches)-1):
            pred_batch_data = pred_data_input[batches[batch_num]:batches[batch_num+1]]  # [-1, 100, 13+5]
            real_len_batch = real_len_input[batches[batch_num]: batches[batch_num+1]] # [-1]

            model_pred = sess.run(pred, feed_dict = {
                    intent: pred_batch_data[:,:,0],
                    click_label_list: pred_batch_data[:,:,2],
                    real_length: real_len_batch,
                    keep_prob: 1.0, # pred
                    like_pxtr_list: pred_batch_data[:,:,13],
                    follow_pxtr_list: pred_batch_data[:,:,14],
                    comment_pxtr_list: pred_batch_data[:,:,15],
                    forward_pxtr_list: pred_batch_data[:,:,16],
                    longview_pxtr_list: pred_batch_data[:,:,17],
                    like_pxtr_dense_list: pred_batch_data[:,:,8],
                    follow_pxtr_dense_list: pred_batch_data[:,:,9],
                    comment_pxtr_dense_list: pred_batch_data[:,:,10],
                    forward_pxtr_dense_list: pred_batch_data[:,:,11],
                    longview_pxtr_dense_list: pred_batch_data[:,:,12]
            })
            pred_list.append(model_pred) # pred = [-1, max_len]

        pred_list = np.concatenate(pred_list, axis=0) # pred_list = [-1, max_len]
        print ("len(pred_list)=", len(pred_list), ", len(pred_data_input)=", len(pred_data_input))

        k = 100
        list_ltr_ndcg_epoch, list_wtr_ndcg_epoch, list_cmtr_ndcg_epoch, list_ftr_ndcg_epoch, list_lvtr_ndcg_epoch = [], [], [], [], []
        ltr_label_ndcg, wtr_label_ndcg, cmtr_label_ndcg, ftr_label_ndcg, lvtr_label_ndcg = [], [], [], [], []
        click_label_ndcg = []
        for i in range(len(pred_list)):
            # pred_list[i] -> [max_len]     pred_data_input[i][:,13] -> [max_len]
            list_ltr_ndcg_epoch.append(ndcg_for_one_samp(pred_data_input[i][:k,13], pred_list[i][:k], k)) # bin
            list_wtr_ndcg_epoch.append(ndcg_for_one_samp(pred_data_input[i][:k,14], pred_list[i][:k], k))
            list_cmtr_ndcg_epoch.append(ndcg_for_one_samp(pred_data_input[i][:k,15], pred_list[i][:k], k))
            list_ftr_ndcg_epoch.append(ndcg_for_one_samp(pred_data_input[i][:k,16], pred_list[i][:k], k))
            list_lvtr_ndcg_epoch.append(ndcg_for_one_samp(pred_data_input[i][:k,17], pred_list[i][:k], k))

            click_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,2], pred_list[i][:k], k))
            ltr_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,3], pred_list[i][:k], k))
            wtr_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,4], pred_list[i][:k], k))
            cmtr_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,5], pred_list[i][:k], k))
            ftr_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,6], pred_list[i][:k], k))
            lvtr_label_ndcg.append(ndcg_for_one_samp(pred_data_input[i][:k,7], pred_list[i][:k], k))
        
        # ndcg: pxtr-input with pred
        print ("[test_data, pxtr-input with pred, ndcg@", k, ", ltr, wtr, cmtr, ftr, lvtr]=", [ 
                sum(list_ltr_ndcg_epoch)/len(list_ltr_ndcg_epoch),
                sum(list_wtr_ndcg_epoch)/len(list_wtr_ndcg_epoch), sum(list_cmtr_ndcg_epoch)/len(list_cmtr_ndcg_epoch),
                sum(list_ftr_ndcg_epoch)/len(list_ftr_ndcg_epoch), sum(list_lvtr_ndcg_epoch)/len(list_lvtr_ndcg_epoch)])
        
        # ndcg: pred with action-label
        print ("[test_data, pred with action-label ndcg@", k, ", click, ltr, wtr, cmtr, ftr, lvtr]=", [
            sum(click_label_ndcg)/len(click_label_ndcg), sum(ltr_label_ndcg)/len(ltr_label_ndcg),
            sum(wtr_label_ndcg)/len(wtr_label_ndcg), sum(cmtr_label_ndcg)/len(cmtr_label_ndcg),
            sum(ftr_label_ndcg)/len(ftr_label_ndcg), sum(lvtr_label_ndcg)/len(lvtr_label_ndcg)])


if __name__ == '__main__':
    print_params(all_para)
    prediction_data(all_para)
    print("pred sample success")