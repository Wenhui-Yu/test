## basic baseline MF_BPR

import tensorflow as tf

class model_MMOE(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'MMOE'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        self.n_users = data['user_num']
        self.num = data['item_num']
        self.max_len = para['ACTION_LIST_MAX_LEN']
        self.loss_weight = para['LOSS_WEIGHT']

        ## placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,), name='users') # index []
        self.items = tf.placeholder(tf.int32, shape=(None,), name='items')
        self.action_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='action_list') # [-1, max_len]
        self.real_length = tf.placeholder(tf.int32, shape=(None,), name='real_length')
        self.label_like = tf.placeholder(tf.int32, shape=(None,), name='label_like')
        self.label_follow = tf.placeholder(tf.int32, shape=(None,), name='label_follow')
        self.label_forward = tf.placeholder(tf.int32, shape=(None,), name='label_forward')

        # print ("self.users=", self.users)
        # print ("self.items=", self.items)
        # print ("self.action_list=", self.action_list)
        # print ("self.real_length=", self.real_length)
        # print ("self.label_like=", self.label_like)
        # print ("self.label_follow=", self.label_follow)
        # print ("self.label_forward=", self.label_forward)

        # reshape
        self.action_list_re = tf.reshape(self.action_list, [-1, self.max_len],)
        self.real_length_re = tf.reshape(self.real_length, [-1, 1])
        self.label_like_re = tf.reshape(self.label_like, [-1, 1], name='label_like_re')
        self.label_follow_re = tf.reshape(self.label_follow, [-1, 1], name='label_follow_re')
        self.label_forward_re = tf.reshape(self.label_forward, [-1, 1], name='label_forward_re')

        ## define trainable parameters
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.random_normal([self.num, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')

        ## lookup
        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users) # [-1, dim]
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items) # [-1, dim]
        self.loss_reg = tf.nn.l2_loss(self.u_embeddings) + tf.nn.l2_loss(self.i_embeddings)

        self.action_list_re = tf.reshape(self.action_list_re, [-1])  # [-1, max_len] -> [bs*max_len]
        self.action_list_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.action_list_re)  # [bs*max_len, dim]
        self.action_list_embeddings = tf.reshape(self.action_list_embeddings, [-1, self.max_len, self.emb_dim])  #[-1, max_len, dim]
        
        # start ---------------------
        mask = tf.sequence_mask(self.real_length_re, maxlen=self.max_len, dtype=tf.float32)
        mask = tf.reshape(mask, [-1, self.max_len]) 

        # target_attention
        self.i_embeddings = tf.reshape(self.i_embeddings, [-1, 1, self.emb_dim])
        # [-1, list_size_q=1, nh*dim]
        taget_attention_input = self.set_attention_block(self.i_embeddings, self.action_list_embeddings, name="target_attention", mask=mask, 
                                col=self.emb_dim, nh=1, action_item_size=self.emb_dim, att_emb_size=self.emb_dim, mask_flag_k=True)
        # print("tf.shape(taget_attention_input)=", tf.shape(taget_attention_input))
        taget_attention_input = tf.reshape(taget_attention_input, [-1, self.emb_dim])

        # mmoe
        self.i_embeddings = tf.reshape(self.i_embeddings, [-1, self.emb_dim])
        feature_input = tf.concat([self.u_embeddings, self.i_embeddings, taget_attention_input], -1)

        feature_input = tf.reshape(feature_input, [-1, 1, self.emb_dim*3])
        # [-1, 1, att_emb_size] ** num_tasks
        mmoe_output = self.mmoe_layer(feature_input, att_emb_size=32, num_experts=6, num_tasks=3)

        # logit
        # [-1, 1, 1]
        like_logit = tf.layers.dense(mmoe_output[0], 1, name='like_predictor_mlp')
        follow_logit = tf.layers.dense(mmoe_output[1], 1, name='follow_predictor_mlp')
        forward_logit = tf.layers.dense(mmoe_output[2], 1, name='forward_predictor_mlp')

        like_logit = tf.reshape(like_logit, [-1, 1])
        follow_logit = tf.reshape(follow_logit, [-1, 1])
        forward_logit = tf.reshape(forward_logit, [-1, 1])

        # pred
        self.like_pred = tf.nn.sigmoid(like_logit, name='like_pred') # [-1, 1]
        self.follow_pred = tf.nn.sigmoid(follow_logit, name='follow_pred')
        self.forward_pred = tf.nn.sigmoid(forward_logit, name='forward_pred')

        self.loss_like = tf.losses.log_loss(self.label_like_re, self.like_pred)
        self.loss_follow = tf.losses.log_loss(self.label_follow_re, self.follow_pred)
        self.loss_forward = tf.losses.log_loss(self.label_forward_re, self.forward_pred)

        self.loss = self.loss_weight[0] * self.loss_like + \
                    self.loss_weight[1] * self.loss_follow + \
                    self.loss_weight[2] * self.loss_forward + \
                    self.lamda * self.loss_reg

        ## optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.updates = self.opt.minimize(self.loss)
        # print("self.updates=", self.updates)

    def inner_product(self, users, items):
        scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
        return scores

    def bpr_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def cross_entropy_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores)) + tf.log(1 - tf.nn.sigmoid(neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list: reg += tf.nn.l2_loss(para)
        return reg

    # query_input =》 [-1, list_size_q=1, dim]     k=v=[-1, list_size_k, dim]
    # retun : [-1, list_size_q=1, nh*dim]
    def set_attention_block(self, query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, mask_flag_k=True):
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


    # inputs = [-1, L, dim]  L=1
    # return = [-1, 1, att_emb_size] ** num_tasks
    def mmoe_layer(self, inputs, att_emb_size=32, num_experts = 1, num_tasks = 1):
        expert_outputs, final_outputs = [], []
        with tf.name_scope('experts_network'):
            for i in range(num_experts):
                expert_layer = tf.layers.dense(inputs, att_emb_size, activation=tf.nn.relu, name='expert{}_'.format(i)+'param')
                expert_outputs.append(tf.expand_dims(expert_layer, axis=3))
        expert_outputs = tf.concat(expert_outputs, 3)  # (batch_size, L, expert_units[-1], num_experts)

        with tf.name_scope('gates_network'):
            for i in range(num_tasks):
                gate_layer = tf.layers.dense(inputs, num_experts, activation=tf.nn.softmax, name='gates{}_'.format(i)+'param')
                expanded_gate_output = tf.expand_dims(gate_layer, 3) # (batch_size, L, num_experts, 1)
                # [-1, L, att, num_experts] * [-1, L, num_experts, 1] = (-1, L, expert_units[-1], 1)
                weighted_expert_output = tf.matmul(expert_outputs, expanded_gate_output) # (batch_size, L, expert_units[-1], 1)
                weighted_expert_output = tf.squeeze(weighted_expert_output, axis=-1)
                final_outputs.append(weighted_expert_output)
        return final_outputs
