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

def read_data1111(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return data, interactions, user_num, item_num

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


def cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_comment_re, epoch_label_forward_re, epoch_label_longview_re,
            epoch_like_pred, epoch_follow_pred, epoch_comment_pred, epoch_forward_pred, epoch_longview_pred):
    list_auc = []
    auc_like, auc_op_like = tf.metrics.auc(tf.concat(epoch_label_like_re, 0), tf.concat(epoch_like_pred, 0))
    auc_follow, auc_op_follow = tf.metrics.auc(tf.concat(epoch_label_follow_re, 0), tf.concat(epoch_follow_pred, 0))
    auc_comment, auc_op_comment = tf.metrics.auc(tf.concat(epoch_label_comment_re, 0), tf.concat(epoch_comment_pred, 0))
    auc_forward, auc_op_forward = tf.metrics.auc(tf.concat(epoch_label_forward_re, 0), tf.concat(epoch_forward_pred, 0))
    auc_longview, auc_op_longview = tf.metrics.auc(tf.concat(epoch_label_longview_re, 0), tf.concat(epoch_longview_pred, 0))
    
    sess.run(tf.local_variables_initializer())
    sess.run([auc_op_like, auc_op_follow, auc_op_comment, auc_op_forward, auc_op_longview])
    auc_like_value, auc_follow_value, auc_comment_value, auc_forward_value, auc_longview_value = sess.run(
        [auc_like, auc_follow, auc_comment, auc_forward, auc_longview])
    
    return [auc_like_value, auc_follow_value, auc_comment_value, auc_forward_value, auc_longview_value]
    # auc_like_value = sess.run(auc_like)
    # list_auc_like_value.append(auc_like_value)

# NOTE: add feature [real_length]
def generate_sample(data, para):
    (user, item, time_ms, click, like, follow, comment, forward, longview, user_real_action) = data
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
    return [user, item, time_ms, click, like, follow, comment, forward, longview, real_length] + limit_user_real_action
