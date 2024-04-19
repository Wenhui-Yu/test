from model_MMOE import *
from utils_mmoe import *
from params import DIR

#    0      1    2      3      4       5         6                7
# [user, item, click, like, follow, forward, real_length] + limit_user_real_action

def train_model(para):
    ## paths of data
    train_path = DIR + 'tenrec_train_data.json'
    validation_path = DIR + 'tenrec_validation_data.json'
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
    batches_validation = list(range(0, len(validation_data), para['BATCH_SIZE']))
    batches_validation.append(len(validation_data))

    ## training iteratively
    list_auc_epoch = []
    list_auc_epoch_vali = []
    F1_max = 0
    for epoch in range(para['N_EPOCH']):
        # train
        epoch_label_like_re, epoch_label_follow_re, epoch_label_forward_re = [], [], []
        epoch_like_pred, epoch_follow_pred, epoch_forward_pred = [], [], []
        for batch_num in range(len(batches)-1):
            train_batch_data = train_data_input[batches[batch_num]: batches[batch_num+1]]

            _, loss, loss_like, loss_follow, loss_forward, \
            label_like_re, label_follow_re, label_forward_re, \
            like_pred, follow_pred, forward_pred = \
            sess.run(
                [model.updates, model.loss, model.loss_like, model.loss_follow, model.loss_forward,
                model.label_like_re, model.label_follow_re, model.label_forward_re,
                model.like_pred, model.follow_pred, model.forward_pred], 
                feed_dict={
                    model.users: train_batch_data[:,0],
                    model.items: train_batch_data[:,1],
                    model.action_list: train_batch_data[:,7:],
                    model.real_length: train_batch_data[:,6],
                    model.label_like: train_batch_data[:,3],
                    model.label_follow: train_batch_data[:,4],
                    model.label_forward: train_batch_data[:,5],
            })
            epoch_label_like_re.append(label_like_re)
            epoch_label_follow_re.append(label_follow_re)
            epoch_label_forward_re.append(label_forward_re)
            epoch_like_pred.append(like_pred)
            epoch_follow_pred.append(follow_pred)
            epoch_forward_pred.append(forward_pred)
        # train auc:
        list_auc = cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_forward_re,
                epoch_like_pred, epoch_follow_pred, epoch_forward_pred)
        list_auc_epoch.append(list_auc)

        # validation:
        epoch_label_like_re_vali, epoch_label_follow_re_vali, epoch_label_forward_re_vali = [], [], []
        epoch_like_pred_vali, epoch_follow_pred_vali, epoch_forward_pred_vali = [], [], []
        for batch_num in range(len(batches_validation)-1):
            validation_batch_data = validation_data_input[batches_validation[batch_num]: batches_validation[batch_num+1]]

            loss_vali, loss_like_vali, loss_follow_vali, loss_forward_vali, \
            label_like_re_vali, label_follow_re_vali, label_forward_re_vali, \
            like_pred_vali, follow_pred_vali, forward_pred_vali = \
            sess.run(
                [model.loss, model.loss_like, model.loss_follow, model.loss_forward,
                model.label_like_re, model.label_follow_re, model.label_forward_re,
                model.like_pred, model.follow_pred, model.forward_pred],
                feed_dict={model.users: validation_batch_data[:, 0],
                        model.items: validation_batch_data[:, 1],
                        model.action_list: validation_batch_data[:, 7:],
                        model.real_length: validation_batch_data[:, 6],
                        model.label_like: validation_batch_data[:, 3],
                        model.label_follow: validation_batch_data[:, 4],
                        model.label_forward: validation_batch_data[:, 5],
            })
            epoch_label_like_re_vali.append(label_like_re_vali)
            epoch_label_follow_re_vali.append(label_follow_re_vali)
            epoch_label_forward_re_vali.append(label_forward_re_vali)
            epoch_like_pred_vali.append(like_pred_vali)
            epoch_follow_pred_vali.append(follow_pred_vali)
            epoch_forward_pred_vali.append(forward_pred_vali)

        list_auc_vali = cal_auc(sess, epoch_label_like_re_vali, epoch_label_follow_re_vali, epoch_label_forward_re_vali,
                                epoch_like_pred_vali, epoch_follow_pred_vali, epoch_forward_pred_vali)
        list_auc_epoch_vali.append(list_auc_vali)

        # model save
        if ((epoch+1) == 30) or ((epoch+1) == 40) or ((epoch+1) == 50) or ((epoch+1) == 60) or ((epoch+1) == para['BEST_EPOCH']):
            print ("start save model , epoch+1=", epoch+1)
            save_path = saver.save(sess, save_model_path, global_step=epoch+1)
            print("model save path = ", save_path)

        # log
        # print_value([epoch + 1, loss, loss_like, loss_follow, loss_forward])
        print("\nTraining auc:")
        print("[epoch + 1, loss, loss_like, loss_follow, loss_forward] = ",
                [epoch + 1, loss, loss_like, loss_follow, loss_forward])
        print("[epoch + 1, like_auc, follow_auc, forward_auc", 
                [epoch + 1, list_auc[0], list_auc[1], list_auc[2]])
        print("\nTest auc:")
        print("[epoch + 1, loss, loss_like, loss_follow, loss_forward] = ",
              [epoch + 1, loss_vali, loss_like_vali, loss_follow_vali, loss_forward_vali])
        print("[epoch + 1, like_auc, follow_auc, forward_auc",
              [epoch + 1, list_auc_vali[0], list_auc_vali[1], list_auc_vali[2]])
        if not loss < 10 ** 10:
            print ("ERROR, loss big, loss=", loss)
            break
    
    # training print auc
    print("\nTraining auc:")
    for epoch in range(len(list_auc_epoch)):
        print("epoch+1=", epoch+1, 
            ", like_auc=", list_auc_epoch[epoch][0], 
            ", follow_auc=", list_auc_epoch[epoch][1], 
            ", forward_auc=", list_auc_epoch[epoch][2])
    # print auc
    print("\nTest auc:")
    for epoch in range(len(list_auc_epoch_vali)):
        print("epoch+1=", epoch + 1,
              ", like_auc=", list_auc_epoch_vali[epoch][0],
              ", follow_auc=", list_auc_epoch_vali[epoch][1],
              ", forward_auc=", list_auc_epoch_vali[epoch][2])
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
