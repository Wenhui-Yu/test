from model_MMOE import *
from utils_mmoe import *
from params import DIR

def train_model(para):
    ## paths of data
    train_path = DIR + 'train_data.json'
    validation_path = DIR + 'validation_data.json'
    save_model_path = './model_ckpt/mmoe_model.ckpt'

    ## Load data
    [train_data, user_num, item_num] = read_data(train_path)
    validation_data = read_data(validation_path)[0]
    print("len(train_data)=",len(train_data), ", user_num=", user_num, ", item_num=", item_num)

    data = {'user_num': user_num, "item_num": item_num}

    ## define the model
    model = model_MMOE(data=data, para=para)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # saver
    saver = tf.train.Saver(max_to_keep = 10)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## process data
    train_data_input = []
    for sample in range(len(train_data)):
        sample_list = generate_sample(train_data[sample], para)
        train_data_input.append(sample_list)
    train_data_input = np.array(train_data_input)
    validation_data_input = []
    for sample in range(len(validation_data)):
        validation_data_input.append(generate_sample(validation_data[sample], para))
    validation_data_input = np.array(validation_data_input)
    ## split the training samples into batches
    batches = list(range(0, len(train_data), para['BATCH_SIZE']))
    batches.append(len(train_data))

    ## training iteratively
    list_auc_epoch = []
    list_auc_epoch_vali = []
    F1_max = 0
    for epoch in range(para['N_EPOCH']):
        epoch_label_like_re, epoch_label_follow_re, epoch_label_comment_re, epoch_label_forward_re, epoch_label_longview_re = [], [], [], [], []
        epoch_like_pred, epoch_follow_pred, epoch_comment_pred, epoch_forward_pred, epoch_longview_pred = [], [], [], [], []
        for batch_num in range(len(batches)-1):
            train_batch_data = train_data_input[batches[batch_num]: batches[batch_num+1]]
            # for sample in range(batches[batch_num], batches[batch_num+1]):
            #     sample_list = generate_sample(train_data[sample], para)
            #     train_batch_data.append(sample_list)
            # train_batch_data = np.array(train_batch_data)
            _, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview, \
            label_like_re, label_follow_re, label_comment_re, label_forward_re, label_longview_re, \
            like_pred, follow_pred, comment_pred, forward_pred, longview_pred = \
            sess.run(
                [model.updates, model.loss, model.loss_like, model.loss_follow, model.loss_comment, 
                    model.loss_forward, model.loss_longview, 
                    model.label_like_re, model.label_follow_re, model.label_comment_re, model.label_forward_re, model.label_longview_re, 
                    model.like_pred, model.follow_pred, model.comment_pred, model.forward_pred, model.longview_pred], 
                feed_dict={model.users: train_batch_data[:,0],
                            model.items: train_batch_data[:,1],
                            model.action_list: train_batch_data[:,10:],
                            model.real_length: train_batch_data[:,9],
                            model.label_like: train_batch_data[:,4],
                            model.label_follow: train_batch_data[:,5],
                            model.label_comment: train_batch_data[:,6],
                            model.label_forward: train_batch_data[:,7],
                            model.label_longview: train_batch_data[:,8],
            })
            epoch_label_like_re.append(label_like_re)
            epoch_label_follow_re.append(label_follow_re)
            epoch_label_comment_re.append(label_comment_re)
            epoch_label_forward_re.append(label_forward_re)
            epoch_label_longview_re.append(label_longview_re)
            epoch_like_pred.append(like_pred)
            epoch_follow_pred.append(follow_pred)
            epoch_comment_pred.append(comment_pred)
            epoch_forward_pred.append(forward_pred)
            epoch_longview_pred.append(longview_pred)
        # train auc:
        list_auc = cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_comment_re, epoch_label_forward_re, epoch_label_longview_re,
                epoch_like_pred, epoch_follow_pred, epoch_comment_pred, epoch_forward_pred, epoch_longview_pred)
        list_auc_epoch.append(list_auc)

        # validation:

        # validation_data_input = []
        # for sample in range(len(validation_data_input)):
        #     validation_data_input.append(generate_sample(validation_data[sample], para))
        # validation_data_input = np.array(validation_data_input)
        loss_vali, loss_like_vali, loss_follow_vali, loss_comment_vali, loss_forward_vali, loss_longview_vali, \
        label_like_re_vali, label_follow_re_vali, label_comment_re_vali, label_forward_re_vali, label_longview_re_vali, \
        like_pred_vali, follow_pred_vali, comment_pred_vali, forward_pred_vali, longview_pred_vali = \
        sess.run(
            [model.loss, model.loss_like, model.loss_follow, model.loss_comment, model.loss_forward, model.loss_longview,
             model.label_like_re, model.label_follow_re, model.label_comment_re, model.label_forward_re, model.label_longview_re,
             model.like_pred, model.follow_pred, model.comment_pred, model.forward_pred, model.longview_pred],
            feed_dict={model.users: validation_data_input[:, 0],
                       model.items: validation_data_input[:, 1],
                       model.action_list: validation_data_input[:, 10:],
                       model.real_length: validation_data_input[:, 9],
                       model.label_like: validation_data_input[:, 4],
                       model.label_follow: validation_data_input[:, 5],
                       model.label_comment: validation_data_input[:, 6],
                       model.label_forward: validation_data_input[:, 7],
                       model.label_longview: validation_data_input[:, 8],
                       })
        list_auc_vali = cal_auc(sess, [label_like_re_vali], [label_follow_re_vali], [label_comment_re_vali], [label_forward_re_vali], [label_longview_re_vali],
                           [like_pred_vali], [follow_pred_vali], [comment_pred_vali], [forward_pred_vali], [longview_pred_vali])
        list_auc_epoch_vali.append(list_auc_vali)

        if ((epoch+1) == para['BEST_EPOCH']):
            print ("start save model , epoch+1=", epoch+1)
            save_path = saver.save(sess, save_model_path, global_step=epoch+1)
            print("model save path = ", save_path)

        # print_value([epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview])
        print("\nTraining auc:")
        print("[epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview] = ",
                [epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview])
        print("[epoch + 1, like_auc, follow_auc, comment_auc, forward_auc, longview_auc", 
                [epoch + 1, list_auc[0], list_auc[1], list_auc[2], list_auc[3], list_auc[4]])
        print("\nTest auc:")
        print("[epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview] = ",
              [epoch + 1, loss_vali, loss_like_vali, loss_follow_vali, loss_comment_vali, loss_forward_vali, loss_longview_vali])
        print("[epoch + 1, like_auc, follow_auc, comment_auc, forward_auc, longview_auc",
              [epoch + 1, list_auc_vali[0], list_auc_vali[1], list_auc_vali[2], list_auc_vali[3], list_auc_vali[4]])
        if not loss < 10 ** 10:
            print ("ERROR, loss big, loss=", loss)
            break
    
    # training print auc
    print("\nTraining auc:")
    for epoch in range(len(list_auc_epoch)):
        print("epoch+1=", epoch+1, 
            ", like_auc=", list_auc_epoch[epoch][0], 
            ", follow_auc=", list_auc_epoch[epoch][1], 
            ", comment_auc=", list_auc_epoch[epoch][2],
            ", forward_auc=", list_auc_epoch[epoch][3],
            ", longview_auc=", list_auc_epoch[epoch][4])
    # print auc
    print("\nTest auc:")
    for epoch in range(len(list_auc_epoch_vali)):
        print("epoch+1=", epoch + 1,
              ", like_auc=", list_auc_epoch_vali[epoch][0],
              ", follow_auc=", list_auc_epoch_vali[epoch][1],
              ", comment_auc=", list_auc_epoch_vali[epoch][2],
              ", forward_auc=", list_auc_epoch_vali[epoch][3],
              ", longview_auc=", list_auc_epoch_vali[epoch][4])
    # save
    # save_path = saver.save(sess, save_model_path)
    # print("model save path = ", save_path)

    #     F1, NDCG = test_model(sess, model, para_test)
    #     if F1[1] > F1_max:
    #         F1_max = F1[1]
    #         user_embeddings, item_embeddings = sess.run([model.user_embeddings, model.item_embeddings])
    #     ## print performance
    #     print_value([epoch + 1, loss, F1_max, F1, NDCG])
    #     if not loss < 10 ** 10:
    #         break
    # save_embeddings([user_embeddings.tolist(), item_embeddings.tolist()], save_embeddings_path)
