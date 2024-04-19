import json
import random as rd
import tensorflow as tf

def print_params(para):
    for para_name in para:
        print(para_name+':  ',para[para_name])

def print_value(value):
    [inter, loss, f1_max, F1, NDCG] = value
    print('iter: %d loss %.2f f1 %.4f' %(inter, loss, f1_max), end='  ')
    print(F1, NDCG)

def save_embeddings(data, path):
    f = open(path, 'w')
    js = json.dumps(data)
    f.write(js)
    f.write('\n')
    f.close

# [[user_id, item_id, click, like, follow, forward, action_list], []]
def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    row_num = len(data)
    print ("sample number=", row_num, ", data_path=", path)
    print("data[:2]=", data[:2])

    user_num = 0;
    item_num = 0;
    for row in data:
        user_num = max(row[0], user_num)
        item_num = max(row[1], item_num)
        
    return data, user_num + 1, item_num + 1


def cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_forward_re,
            epoch_like_pred, epoch_follow_pred, epoch_forward_pred):
    list_auc = []
    auc_like, auc_op_like = tf.metrics.auc(tf.concat(epoch_label_like_re, 0), tf.concat(epoch_like_pred, 0))
    auc_follow, auc_op_follow = tf.metrics.auc(tf.concat(epoch_label_follow_re, 0), tf.concat(epoch_follow_pred, 0))
    auc_forward, auc_op_forward = tf.metrics.auc(tf.concat(epoch_label_forward_re, 0), tf.concat(epoch_forward_pred, 0))
    
    sess.run(tf.local_variables_initializer())
    sess.run([auc_op_like, auc_op_follow, auc_op_forward])
    auc_like_value, auc_follow_value, auc_forward_value = sess.run(
        [auc_like, auc_follow, auc_forward])
    
    return [auc_like_value, auc_follow_value, auc_forward_value]
    # auc_like_value = sess.run(auc_like)
    # list_auc_like_value.append(auc_like_value)

# NOTE: add feature [real_length]
def generate_sample(data, para):
    (user, item, click, like, follow, forward, user_real_action) = data
    limit_user_real_action = user_real_action
    cur_length = len(limit_user_real_action)
    real_length = 0
    if cur_length >= para['ACTION_LIST_MAX_LEN']:
        limit_user_real_action = limit_user_real_action[-para['ACTION_LIST_MAX_LEN']:]  # tail
        real_length = len(limit_user_real_action)
    else:
        real_length = len(limit_user_real_action)
        list_null_pos = []
        for i in range(para['ACTION_LIST_MAX_LEN'] - cur_length):
            list_null_pos.append(0)
        limit_user_real_action = limit_user_real_action + list_null_pos # first item_id, then 0; use cal attention with mask
    # print("len(limit_user_real_action)=", len(limit_user_real_action), ", real_length=", real_length)
    # print("limit_user_real_action=", limit_user_real_action)
    return [user, item, click, like, follow, forward, real_length] + limit_user_real_action
