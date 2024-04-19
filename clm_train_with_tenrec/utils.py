import json
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn

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


# [
#           0        1             2     3     4         5      6    7      8
#     [ item_id, index_time_ms, click, like, follow, forward, pltr, pwtr, pftr],  [] .....]
#     []
# ....
# ]
def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    row_num = len(data)
    print ("sample number=", row_num, ", data_path=", path)

    item_num = 0
    for sample in data:
        for item in sample:
            item_num = max(item[0], item_num)
    
    print ("item_num=", item_num+1) # 25701
    return data, item_num + 1


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


def generate_sample_with_max_len(data, para):
    sample = data
    real_len = len(sample)
    # print ("real_len=", real_len)
    
    # NOTE: real_len <= para['CANDIDATE_ITEM_LIST_LENGTH']
    if real_len == para['CANDIDATE_ITEM_LIST_LENGTH']:
        return sample, real_len
    elif (real_len < para['CANDIDATE_ITEM_LIST_LENGTH']):
        for i in range(para['CANDIDATE_ITEM_LIST_LENGTH'] - real_len):
            sample.append([0,0,0,0,0,0,0,0,0])
    return sample, real_len

def generate_sample_with_pxtr_bins(data, para, pxtr_bucket_range):
    sample = []
    for (item_id, index_time_ms, click, like, follow, forward, pltr, pwtr, pftr) in data:
        pltr = max(pltr ,0.00000000)
        pltr = min(pltr ,0.99999999)
        pwtr = max(pwtr ,0.00000000)
        pwtr = min(pwtr ,0.99999999)
        pftr = max(pftr ,0.00000000)
        pftr = min(pftr ,0.99999999)

        pltr_index = np.searchsorted(pxtr_bucket_range, pltr)
        pwtr_index = np.searchsorted(pxtr_bucket_range, pwtr)
        pftr_index = np.searchsorted(pxtr_bucket_range, pftr)
        
        sample.append([item_id, index_time_ms, click, like, follow, forward, pltr, pwtr, pftr,
                    pltr_index, pwtr_index, pftr_index])
    return sample

def get_order(ranking):
    position_in_ranking = np.argsort(-np.array(ranking))
    order = np.argsort(position_in_ranking) + 1
    return order

def ndcg_for_one_samp(ranking_xtr, ranking_ens, k):
    ranking_xtr = ranking_xtr[:k]
    ranking_ens = ranking_ens[:k]
    order_xtr = get_order(ranking_xtr)
    order_ens = get_order(ranking_ens)
    dcg, idcg = 0, 0
    for i in range(len(ranking_xtr)):
        dcg += ranking_xtr[i] / np.log(order_ens[i] + 1) / np.log(2.0)
        idcg += ranking_xtr[i] / np.log(order_xtr[i] + 1) / np.log(2.0)
    return dcg / (idcg + 1e-10)

def auc_for_one_samp(ranking_xtr, ranking_ens, k):
    ranking_xtr = ranking_xtr[:k]
    ranking_ens = ranking_ens[:k]
    pos_index = []
    neg_index = []
    if ranking_xtr[0] == 0 or ranking_xtr[0] == 1:
        pos_index = np.where(ranking_xtr > 0.5)
        neg_index = np.where(ranking_xtr < 0.5)
    if ranking_xtr[0] != 0 and ranking_xtr[0] != 1:
        mean = np.mean(ranking_xtr)
        pos_index = np.where(ranking_xtr > mean)
        neg_index = np.where(ranking_xtr < mean)
    corr_count = 0
    total_count = 0
    for p_id in pos_index[0]:
        for n_id in neg_index[0]:
            total_count += 1
            if ranking_ens[p_id] - ranking_ens[n_id] > 0:
                corr_count += 1
    return corr_count / max(total_count, 1e-10)

def map_for_one_samp(ranking_xtr, ranking_ens, k):
    ranking_xtr = ranking_xtr[:k]
    ranking_ens = ranking_ens[:k]
    a, b = zip(*sorted(zip(ranking_xtr, ranking_ens), key=lambda x: x[1], reverse=True))
    num_pos = 0
    ap_acc = 0
    for i in range(k):
        if a[i] == 1:
            num_pos += 1
            ap = num_pos / (i + 1)
            ap_acc += ap
    return ap_acc / max(num_pos, 1e-10)

# metrices with @k
def evaluation_F1(order, top_k, positive_item):
    epsilon = 0.1 ** 10
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / max(len(top_k_items), epsilon)
    recall = len(top_k_items & positive_item) / max(len(positive_item), epsilon)
    F1 = 2 * precision * recall / max(precision + recall, epsilon)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    epsilon = 0.1**10
    DCG = 0
    iDCG = 0
    for i in range(top_k):
        if top_k_item[i] in positive_item:
            DCG += 1 / np.log2(i + 2)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / np.log2(i + 2)
    NDCG = DCG / max(iDCG, epsilon)
    return NDCG

def print_pxtr_ndcg(epoch, train_data_input, pred_list, train_test):
    # ndcg
    k = 100
    list_ltr_ndcg_epoch, list_wtr_ndcg_epoch, list_ftr_ndcg_epoch = [], [], []
    ltr_label_ndcg, wtr_label_ndcg, ftr_label_ndcg = [], [], []
    click_label_ndcg = []
    for i in range(len(pred_list)):  #len(pred_list)):
        # pred_list[i]     [max_len]
        # train_data_input[i]->[max_len, 13+5]      train_data_input[i][:,13] # [max_len]
        list_ltr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,9], pred_list[i][:k], k)) # bin
        list_wtr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,10], pred_list[i][:k], k))
        list_ftr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,11], pred_list[i][:k], k))

        click_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,2], pred_list[i][:k], k))
        ltr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,3], pred_list[i][:k], k))
        wtr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,4], pred_list[i][:k], k))
        ftr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,5], pred_list[i][:k], k))
    print (train_test, "ep", epoch+1,
        "%.4f"%(sum(click_label_ndcg)/len(click_label_ndcg)), "%.4f"%(sum(ltr_label_ndcg)/len(ltr_label_ndcg)),
        "%.4f"%(sum(wtr_label_ndcg)/len(wtr_label_ndcg)), "%.4f"%(sum(ftr_label_ndcg)/len(ftr_label_ndcg)), end='   ')
    print ("%.4f"%(sum(list_ltr_ndcg_epoch)/len(list_ltr_ndcg_epoch)), "%.4f"%(sum(list_wtr_ndcg_epoch)/len(list_wtr_ndcg_epoch)),
        "%.4f"%(sum(list_ftr_ndcg_epoch)/len(list_ftr_ndcg_epoch)), end='   ')

def print_pxtr_auc(epoch, train_data_input, pred_list, train_test):
    # auc
    k = 100
    list_ltr_auc_epoch, list_wtr_auc_epoch, list_ftr_auc_epoch = [], [], []
    ltr_label_auc, wtr_label_auc, ftr_label_auc = [], [], []
    click_label_auc = []
    for i in range(len(pred_list)):  # len(pred_list)):
        # pred_list[i]     [max_len]
        # train_data_input[i]->[max_len, 13+5]      train_data_input[i][:,13] # [max_len]
        list_ltr_auc_epoch.append(auc_for_one_samp(train_data_input[i][:k, 9], pred_list[i][:k], k))  # bin
        list_wtr_auc_epoch.append(auc_for_one_samp(train_data_input[i][:k, 10], pred_list[i][:k], k))
        list_ftr_auc_epoch.append(auc_for_one_samp(train_data_input[i][:k, 11], pred_list[i][:k], k))

        click_label_auc.append(auc_for_one_samp(train_data_input[i][:k, 2], pred_list[i][:k], k))
        ltr_label_auc.append(auc_for_one_samp(train_data_input[i][:k, 3], pred_list[i][:k], k))
        wtr_label_auc.append(auc_for_one_samp(train_data_input[i][:k, 4], pred_list[i][:k], k))
        ftr_label_auc.append(auc_for_one_samp(train_data_input[i][:k, 5], pred_list[i][:k], k))
    print("%.4f" % (sum(click_label_auc) / len(click_label_auc)),
          "%.4f" % (sum(ltr_label_auc) / len(ltr_label_auc)),
          "%.4f" % (sum(wtr_label_auc) / len(wtr_label_auc)), "%.4f" % (sum(ftr_label_auc) / len(ftr_label_auc)), end='   ')
    print("%.4f" % (sum(list_ltr_auc_epoch) / len(list_ltr_auc_epoch)),
          "%.4f" % (sum(list_wtr_auc_epoch) / len(list_wtr_auc_epoch)),
          "%.4f" % (sum(list_ftr_auc_epoch) / len(list_ftr_auc_epoch)), end='   ')

def print_pxtr_map(epoch, train_data_input, pred_list, train_test):
    # map
    k = 100
    list_ltr_map_epoch, list_wtr_map_epoch, list_ftr_map_epoch = [], [], []
    ltr_label_map, wtr_label_map, ftr_label_map = [], [], []
    click_label_map = []
    for i in range(len(pred_list)):  # len(pred_list)):
        # pred_list[i]     [max_len]
        # train_data_input[i]->[max_len, 13+5]      train_data_input[i][:,13] # [max_len]
        # list_ltr_map_epoch.append(map_for_one_samp(train_data_input[i][:k, 9], pred_list[i][:k], k))  # bin
        # list_wtr_map_epoch.append(map_for_one_samp(train_data_input[i][:k, 10], pred_list[i][:k], k))
        # list_ftr_map_epoch.append(map_for_one_samp(train_data_input[i][:k, 11], pred_list[i][:k], k))

        click_label_map.append(map_for_one_samp(train_data_input[i][:k, 2], pred_list[i][:k], k))
        ltr_label_map.append(map_for_one_samp(train_data_input[i][:k, 3], pred_list[i][:k], k))
        wtr_label_map.append(map_for_one_samp(train_data_input[i][:k, 4], pred_list[i][:k], k))
        ftr_label_map.append(map_for_one_samp(train_data_input[i][:k, 5], pred_list[i][:k], k))
    print("%.4f" % (sum(click_label_map) / len(click_label_map)),
          "%.4f" % (sum(ltr_label_map) / len(ltr_label_map)),
          "%.4f" % (sum(wtr_label_map) / len(wtr_label_map)), "%.4f" % (sum(ftr_label_map) / len(ftr_label_map)))
    # print("%.4f" % (sum(list_ltr_map_epoch) / len(list_ltr_map_epoch)),
    #       "%.4f" % (sum(list_wtr_map_epoch) / len(list_wtr_map_epoch)),
    #       "%.4f" % (sum(list_ftr_map_epoch) / len(list_ftr_map_epoch)))

def print_click_ndcg(epoch, top_k, train_data_input, pred_list, train_test):
    f1score = []
    ndcg = []
    for i in range(len(top_k)):
        f1score.append([])
        ndcg.append([])
    for i in range(len(top_k)):
        for j in range(len(pred_list)):
            k = top_k[i]
            pos_items = np.where(train_data_input[j][:, 2] > 0.5)[0]
            topk_items = np.argsort(-pred_list[j])[:k]
            f1score[i].append(evaluation_F1(topk_items, k, pos_items))
            ndcg[i].append(evaluation_NDCG(topk_items, k, pos_items))
    # ndcg: pred with action-label
    f1score = np.array(f1score)
    ndcg = np.array(ndcg)
    print (train_test, "ep", epoch+1, np.mean(f1score, 1), np.mean(ndcg, 1))

def print_loss(epoch, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias):
    print("[epoch+1, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias] = ",
          [epoch+1, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias])
def save_ckpt(epoch, sess, saver, save_model_path):
    if ((epoch+1) == 5) or ((epoch+1) == 10):
        print ("start save model , epoch+1=", epoch+1)
        save_path = saver.save(sess, save_model_path, global_step=epoch+1)

########################################## model utils ##############################################
########################################## model utils ##############################################

def linear_set_attention_block(query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, m_size=32, iter_num=0):
    ## poly encoder
    with tf.name_scope(name):
        I = tf.get_variable(name + "_i_trans_matrix",(1, m_size, col), initializer=tf.truncated_normal_initializer(stddev=5.0)) # [-1, m_size, col]
        I = tf.tile(I, [tf.shape(query_input)[0],1,1])
        H = set_attention_block(I, action_list_input, name + "_ele2clus", mask, col, 1, action_item_size, att_emb_size, True, True)    #[-1, m_size, nh*dim]
        H_list = [H]
        for l in range(iter_num):
            H += set_attention_block(H, H, name + "_sa_clus2clus_{}".format(l), mask, att_emb_size, 1, att_emb_size, att_emb_size, False, False)
            H = CommonLayerNorm(H, scope='ln1_clus2clus_{}'.format(l))
            H += tf.layers.dense(tf.nn.relu(H), att_emb_size, name='ffn_clus2clus_{}'.format(l))
            H = CommonLayerNorm(H, scope='ln2_clus2clus_{}'.format(l))
            H_list.append(H)
        H = tf.reduce_sum(H_list, axis=0)
        res = set_attention_block(query_input, H, name + "_clus2ele", mask, col, nh, att_emb_size, att_emb_size, True, False)
    return res

# query_input =》 [-1, list_size_q=1, dim]     k=v=[-1, list_size_k, dim]
# retun : [-1, list_size_q=1, nh*dim]
def set_attention_block(query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, if_mask=True, mask_flag_k=True):
    with tf.name_scope("mha_" + name):
        batch_size = tf.shape(query_input)[0]
        list_size = tf.shape(query_input)[1]
        list_size_k = tf.shape(action_list_input)[1]
        Q = tf.get_variable(name + '_q_trans_matrix', (col, att_emb_size * nh))
        K = tf.get_variable(name + '_k_trans_matrix', (action_item_size, att_emb_size * nh))
        V = tf.get_variable(name + '_v_trans_matrix', (action_item_size, att_emb_size * nh))

        querys = tf.tensordot(query_input, Q, axes=(-1, 0))
        keys = tf.tensordot(action_list_input, K, axes=(-1, 0))
        values = tf.tensordot(action_list_input, V, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, nh, axis=2))
        keys = tf.stack(tf.split(keys, nh, axis=2))
        values = tf.stack(tf.split(values, nh, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True) / 8.0
        if if_mask:
            trans_mask = tf.tile(tf.expand_dims(mask, axis=0),[nh, 1, 1])
            if mask_flag_k: trans_mask = tf.tile(tf.expand_dims(trans_mask, axis=2), [1,1,list_size,1])
            else: trans_mask = tf.tile(tf.expand_dims(trans_mask, axis=3), [1, 1, 1, list_size_k])
            paddings = tf.ones_like(trans_mask) * (-2 ** 32 + 1)
            inner_product = tf.where(tf.equal(trans_mask, 0), paddings, inner_product)

        normalized_att_scores = tf.nn.softmax(inner_product)
        result = tf.matmul(normalized_att_scores, values)
        result = tf.transpose(result, perm=[1, 2, 0, 3])
        mha_result = tf.reshape(result, [batch_size, list_size, nh * att_emb_size])
    return mha_result


def CommonLayerNorm(inputs,
                center=True,
                scale=True,
                activation_fn=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                begin_norm_axis=1,
                begin_params_axis=-1,
                scope=None):
    with variable_scope.variable_scope(
            scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                        'must be < rank(inputs) (%d)' %
                        (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                                'beta')
            beta = tf.get_variable(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.zeros_initializer(),
                collections=beta_collections,
                trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(
                variables_collections, 'gamma')
            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=gamma_collections,
                trainable=trainable)
        # Calculate the moments on the last axis (layer activations).
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1e-12
        outputs = nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def add_position_emb(query_input, pxtr_dense, seq_length, pxtr_num, dim, decay, name):
    with tf.name_scope(name):
        pos_embeddings = tf.get_variable(name+"_pos_embeddings", (seq_length, dim), initializer=tf.truncated_normal_initializer(stddev=5.0))  # [-1, m_size, col]
        position_in_ranking = tf.contrib.framework.argsort(tf.stop_gradient(pxtr_dense), axis=1)
        order = tf.contrib.framework.argsort(position_in_ranking, axis=1)
        pos_emb = tf.gather(pos_embeddings, order)
        return decay * tf.reshape(pos_emb, [-1, seq_length, pxtr_num * dim]) + query_input
        # return tf.concat([tf.reshape(pos_emb, [-1, seq_length, pxtr_num * dim]), query_input], -1)

def pxtr_transformer(pxtr_input, listwise_len, pxtr_num, dim, name):
    pxtr_input = tf.reshape(pxtr_input, [-1, pxtr_num, dim])
    pxtr_input = set_attention_block(pxtr_input, pxtr_input, name + "_pxtr_transformer", 0, dim, 1, dim, dim, False, False)
    return tf.reshape(pxtr_input, [-1, listwise_len, pxtr_num * dim])

def sigmoid(x):
    return 2 * tf.nn.sigmoid(x) - 1

def layer_dense(pxtr_input, num):
    weight = tf.layers.dense(pxtr_input, num, name='weight')
    return weight

def sim_order_reg_core(seq_1, seq_2, if_norm, length):
    seq_conc = tf.concat([tf.expand_dims(seq_1, -1), tf.expand_dims(seq_2, -1)], -1)
    seq_cut = seq_conc[:, 0: length, :]
    random_index = tf.random.uniform((1, 200), minval=0, maxval=tf.cast(length, dtype=tf.float32))
    random_index = tf.squeeze(tf.cast(tf.floor(random_index), dtype=tf.int64))
    seq_samp = tf.gather(seq_cut, random_index, axis=1)
    seq = tf.reshape(seq_samp, [-1, 2])
    if if_norm:
        seq_mean, seq_var = tf.nn.moments(tf.stop_gradient(seq), axes=0)
        seq_norm = (seq - tf.expand_dims(seq_mean, 0)) / (tf.sqrt(tf.expand_dims(seq_var, 0)) + 0.1 ** 10)
    else:
        seq_norm = seq
    seq_resh = tf.reshape(seq_norm, [-1, 2, 2])
    # attention! sigmoid(x) = 2 * tf.nn.sigmoid(x) - 1
    # TODO: replace with tanh(x)
    reg_loss = tf.multiply(sigmoid(seq_resh[:, 0, 0] - seq_resh[:, 1, 0]), sigmoid(seq_resh[:, 0, 1] - seq_resh[:, 1, 1]))
    return -tf.reduce_mean(reg_loss)

def sim_order_reg(pred, pxtr, weight, length):
    reg_loss = 0
    for i, w in enumerate(weight):
        reg_loss += w * sim_order_reg_core(pred, pxtr[:, :, i], True, length)
        # reg_loss += weaken_bad_pxtr_weight * w * sim_order_reg_core(pred, -1 / (pxtr[:, :, i] + 0.1 ** 10), True, length)
    return reg_loss