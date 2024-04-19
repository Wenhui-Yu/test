from params import all_para
from params import DIR
from utils_mmoe import *

import json
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']

def mmoe_prediction_data(para):
    pred_data_path = DIR + 'train_data_pred.json'
    ltr_data_path = DIR + 'kuairand_ltr_data.json'

    model_path = 'model_ckpt/mmoe_model.ckpt-{}.meta'.format(all_para['BEST_EPOCH'])
    restore_path = 'model_ckpt/mmoe_model.ckpt-{}'.format(all_para['BEST_EPOCH'])

    ## Load data
    pred_data, _, _ = read_data(pred_data_path)
    print ("pred_data[0:3]=", pred_data[0:3])

    ## split the pred-samples into batches
    batches = list(range(0, len(pred_data), para['PRED_BATCH_SIZE']))
    batches.append(len(pred_data))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, restore_path)

        # feed_dict
        user = sess.graph.get_tensor_by_name('users:0')
        item = sess.graph.get_tensor_by_name('items:0')
        action_list = sess.graph.get_tensor_by_name('action_list:0')
        real_length = sess.graph.get_tensor_by_name('real_length:0')
        label_like = sess.graph.get_tensor_by_name('label_like:0')
        label_follow = sess.graph.get_tensor_by_name('label_follow:0')
        label_comment = sess.graph.get_tensor_by_name('label_comment:0')
        label_forward = sess.graph.get_tensor_by_name('label_forward:0')
        label_longview = sess.graph.get_tensor_by_name('label_longview:0')

        # loss  sess.graph.get_tensor_by_name('')
        loss_like = sess.graph.get_tensor_by_name('log_loss/value:0')
        loss_follow = sess.graph.get_tensor_by_name('log_loss_1/value:0')
        loss_comment = sess.graph.get_tensor_by_name('log_loss_2/value:0')
        loss_forward = sess.graph.get_tensor_by_name('log_loss_3/value:0')
        loss_longview = sess.graph.get_tensor_by_name('log_loss_4/value:0')
        loss = sess.graph.get_tensor_by_name('add_3:0')

        # label: cal auc
        label_like_re = sess.graph.get_tensor_by_name('label_like_re:0')
        label_follow_re = sess.graph.get_tensor_by_name('label_follow_re:0')
        label_comment_re = sess.graph.get_tensor_by_name('label_comment_re:0')
        label_forward_re = sess.graph.get_tensor_by_name('label_forward_re:0')
        label_longview_re = sess.graph.get_tensor_by_name('label_longview_re:0')

        # pred: cal auc & save
        like_pred = sess.graph.get_tensor_by_name('like_pred:0')
        follow_pred = sess.graph.get_tensor_by_name('follow_pred:0')
        comment_pred = sess.graph.get_tensor_by_name('comment_pred:0')
        forward_pred = sess.graph.get_tensor_by_name('forward_pred:0')
        longview_pred = sess.graph.get_tensor_by_name('longview_pred:0')

        # opt
        # updates = sess.graph.get_operation_by_name('GradientDescent/GradientDescent/-apply')  # NOTE: SGD

        # for
        epoch_label_like_re = []
        epoch_label_follow_re = []
        epoch_label_comment_re = []
        epoch_label_forward_re = []
        epoch_label_longview_re = []
        epoch_like_pred = []
        epoch_follow_pred = []
        epoch_comment_pred = []
        epoch_forward_pred = []
        epoch_longview_pred = []
        for batch_num in range(len(batches)-1):
            pred_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                sample_list = generate_sample(pred_data[sample], para)
                pred_batch_data.append(sample_list)
            pred_batch_data = np.array(pred_batch_data)


            model_loss, model_loss_like, model_loss_follow, model_loss_comment, model_loss_forward, model_loss_longview, \
            model_label_like_re, model_label_follow_re, model_label_comment_re, model_label_forward_re, model_label_longview_re, \
            model_like_pred, model_follow_pred, model_comment_pred, model_forward_pred, model_longview_pred = \
            sess.run(
                [loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview,
                 label_like_re, label_follow_re, label_comment_re, label_forward_re, label_longview_re,
                 like_pred, follow_pred, comment_pred, forward_pred, longview_pred],
                feed_dict = {
                    user: pred_batch_data[:,0],
                    item: pred_batch_data[:,1],
                    action_list: pred_batch_data[:,10:],
                    real_length: pred_batch_data[:,9],
                    label_like: pred_batch_data[:,4],
                    label_follow: pred_batch_data[:,5],
                    label_comment: pred_batch_data[:,6],
                    label_forward: pred_batch_data[:,7],
                    label_longview: pred_batch_data[:,8],
            })

            epoch_label_like_re.append(model_label_like_re)
            epoch_label_follow_re.append(model_label_follow_re)
            epoch_label_comment_re.append(model_label_comment_re)
            epoch_label_forward_re.append(model_label_forward_re)
            epoch_label_longview_re.append(model_label_longview_re)
            epoch_like_pred.append(model_like_pred)
            epoch_follow_pred.append(model_follow_pred)
            epoch_comment_pred.append(model_comment_pred)
            epoch_forward_pred.append(model_forward_pred)
            epoch_longview_pred.append(model_longview_pred)

            if batches[batch_num] % 20000 == 0:
                # print ("model_like_pred=", model_like_pred)
                print("[batch_start, batch_end, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview] = ", 
                [batches[batch_num], batches[batch_num+1], model_loss,
                model_loss_like, model_loss_follow, model_loss_comment, model_loss_forward, model_loss_longview])
        
        list_auc = cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_comment_re, epoch_label_forward_re, epoch_label_longview_re,
                epoch_like_pred, epoch_follow_pred, epoch_comment_pred, epoch_forward_pred, epoch_longview_pred)
        print("pred_data AUC", 
            ", like_auc=", list_auc[0], 
            ", follow_auc=", list_auc[1], 
            ", comment_auc=", list_auc[2],
            ", forward_auc=", list_auc[3],
            ", longview_auc=", list_auc[4])

     
        like_pxtr = [pxtr for batch_list in epoch_like_pred for pxtr in batch_list]
        follow_pxtr = [pxtr for batch_list in epoch_follow_pred for pxtr in batch_list]
        comment_pxtr = [pxtr for batch_list in epoch_comment_pred for pxtr in batch_list]
        forward_pxtr = [pxtr for batch_list in epoch_forward_pred for pxtr in batch_list]
        longview_pxtr = [pxtr for batch_list in epoch_longview_pred for pxtr in batch_list]

        # generate ltr model train data
        ltr_train_data = []
        index = 0
        for (user, item, time_ms, click, like, follow, comment, forward, longview, user_real_action) in pred_data:
            ltr_train_data.append([user, item, time_ms, click, like, follow, comment, forward, longview,
                                round(float(like_pxtr[index]),8), round(float(follow_pxtr[index]),8), round(float(comment_pxtr[index]),8),
                                round(float(forward_pxtr[index]),8), round(float(longview_pxtr[index]),8)])
            index = index + 1  
        json_ltr_train_data = json.dumps(ltr_train_data)
        with open(ltr_data_path, 'w') as file:
            file.write(json_ltr_train_data)
        file.close()
        
        # test: read
        ltr_train_data_v1, _, _ = read_data(ltr_data_path)



if __name__ == '__main__':
    print_params(all_para)
    mmoe_prediction_data(all_para)
    print("pred sample && generate ltr_train_data.json success")
