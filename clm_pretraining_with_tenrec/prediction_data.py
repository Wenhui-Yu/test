from params import all_para
from params import DIR
from utils_mmoe import *

import json
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']

def mmoe_prediction_data(para):
    pred_data_path = DIR + 'tenrec_pred_data.json'
    ltr_data_path = DIR + 'tenrec_ltr_data.json'

    model_path = 'model_ckpt/mmoe_model.ckpt-{}.meta'.format(all_para['BEST_EPOCH'])
    restore_path = 'model_ckpt/mmoe_model.ckpt-{}'.format(all_para['BEST_EPOCH'])
    # model_path = 'model_ckpt/mmoe_model.ckpt-{}.meta'.format('5')
    # restore_path = 'model_ckpt/mmoe_model.ckpt-{}'.format('5')

    ## Load data
    pred_data, _, _ = read_data(pred_data_path)
    print ("pred_data[0:3]=", pred_data[0:3])

    ## split the pred-samples into batches
    batches = list(range(0, len(pred_data), para['PRED_BATCH_SIZE']))
    batches.append(len(pred_data))
    print ("here-0.1")
    with tf.Session() as sess:
        print ("here-1")
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, restore_path)

        print ("here-0.2")
        # feed_dict
        user = sess.graph.get_tensor_by_name('users:0')
        item = sess.graph.get_tensor_by_name('items:0')
        action_list = sess.graph.get_tensor_by_name('action_list:0')
        real_length = sess.graph.get_tensor_by_name('real_length:0')
        label_like = sess.graph.get_tensor_by_name('label_like:0')
        label_follow = sess.graph.get_tensor_by_name('label_follow:0')
        label_forward = sess.graph.get_tensor_by_name('label_forward:0')

        # loss  sess.graph.get_tensor_by_name('')
        loss_like = sess.graph.get_tensor_by_name('log_loss/value:0')
        loss_follow = sess.graph.get_tensor_by_name('log_loss_1/value:0')
        loss_forward = sess.graph.get_tensor_by_name('log_loss_2/value:0')
        loss = sess.graph.get_tensor_by_name('add_3:0')

        # label: cal auc
        label_like_re = sess.graph.get_tensor_by_name('label_like_re:0')
        label_follow_re = sess.graph.get_tensor_by_name('label_follow_re:0')
        label_forward_re = sess.graph.get_tensor_by_name('label_forward_re:0')

        # pred: cal auc & save
        like_pred = sess.graph.get_tensor_by_name('like_pred:0')
        follow_pred = sess.graph.get_tensor_by_name('follow_pred:0')
        forward_pred = sess.graph.get_tensor_by_name('forward_pred:0')
        print ("here-2")

        # opt
        # updates = sess.graph.get_operation_by_name('GradientDescent/GradientDescent/-apply')  # NOTE: SGD
        # updates = sess.graph.get_operation_by_name('Adam/Adam/-apply')  # NOTE: Adam

        # for
        epoch_label_like_re, epoch_label_follow_re, epoch_label_forward_re = [], [], []
        epoch_like_pred, epoch_follow_pred, epoch_forward_pred = [], [], []
        for batch_num in range(len(batches)-1):
            pred_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                sample_list = generate_sample(pred_data[sample], para)
                pred_batch_data.append(sample_list)
            pred_batch_data = np.array(pred_batch_data)
            print ("here-3")


            model_loss, model_loss_like, model_loss_follow, model_loss_forward, \
            model_label_like_re, model_label_follow_re, model_label_forward_re, \
            model_like_pred, model_follow_pred, model_forward_pred = \
            sess.run(
                [loss, loss_like, loss_follow, loss_forward,
                 label_like_re, label_follow_re, label_forward_re,
                 like_pred, follow_pred, forward_pred],
                feed_dict = {
                    user: pred_batch_data[:,0],
                    item: pred_batch_data[:,1],
                    action_list: pred_batch_data[:,7:],
                    real_length: pred_batch_data[:,6],
                    label_like: pred_batch_data[:,3],
                    label_follow: pred_batch_data[:,4],
                    label_forward: pred_batch_data[:,5],
            })

            epoch_label_like_re.append(model_label_like_re)
            epoch_label_follow_re.append(model_label_follow_re)
            epoch_label_forward_re.append(model_label_forward_re)
            epoch_like_pred.append(model_like_pred)
            epoch_follow_pred.append(model_follow_pred)
            epoch_forward_pred.append(model_forward_pred)

            if batches[batch_num] % 1000000 == 0:
                # print ("model_like_pred=", model_like_pred)
                print("[batch_start, batch_end, loss, loss_like, loss_follow, loss_forward] = ", 
                [batches[batch_num], batches[batch_num+1], model_loss,
                model_loss_like, model_loss_follow, model_loss_forward])
        
        list_auc = cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_forward_re,
                epoch_like_pred, epoch_follow_pred, epoch_forward_pred)
        print("pred_data AUC", 
            ", like_auc=", list_auc[0], 
            ", follow_auc=", list_auc[1], 
            ", forward_auc=", list_auc[2])

     
        like_pxtr = [pxtr for batch_list in epoch_like_pred for pxtr in batch_list]
        follow_pxtr = [pxtr for batch_list in epoch_follow_pred for pxtr in batch_list]
        forward_pxtr = [pxtr for batch_list in epoch_forward_pred for pxtr in batch_list]

        # generate ltr model train data
        ltr_train_data = []
        index = 0
        for (user, item, click, like, follow, forward, user_real_action) in pred_data:
            ltr_train_data.append([user, item, click, like, follow, forward,
                                round(float(like_pxtr[index]),8), round(float(follow_pxtr[index]),8), 
                                round(float(forward_pxtr[index]),8)])
            index = index + 1  
        json_ltr_train_data = json.dumps(ltr_train_data)
        with open(ltr_data_path, 'w') as file:
            file.write(json_ltr_train_data)
        file.close()
        
        print("model read data -> pred_data_path=", pred_data_path)
        print("model load path -> restore_path=", restore_path)
        print("model pred save data -> ltr_data_path=", ltr_data_path)
        # test: read
        ltr_train_data_v1, _, _ = read_data(ltr_data_path)



if __name__ == '__main__':
    print_params(all_para)
    mmoe_prediction_data(all_para)
    print("pred sample && generate ltr_train_data.json success")
