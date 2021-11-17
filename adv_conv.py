'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------


'''
# from module import *
import os
from Code.module import *

def model_base(C, E, Y,BA,DR, DISCRIMINATOR_BA, DISCRIMINATOR_DR):
    for gen_idx in range(GEN_NUM):
        if gen_idx == 0:
            layer = tf.reshape(FC_model(CNN_model(C[gen_idx], BA), E[gen_idx], BA, DR), [1, -1])  # 마지막에 conv 에서는 timestamp
        else:
            layer = tf.concat([layer, tf.reshape(FC_model(CNN_model(C[gen_idx], BA, True), E[gen_idx], BA, DR, True), [1, -1])], axis=0)

    Y = tf.reshape(Y, [12, -1]) #일단 통일시켜놨기 떄문에 어쩔 수 없는 부분
    # 3차원 오차, MAE, MAPE는 Train 에서는 필요 없음
    # cost_MAE = MAE(Y, layer)
    train_MSE = MSE(Y, layer)
    # cost_MAPE = MAPE(Y, layer)
    cost_MAE = MAE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1]) #실제로는 직후부터 60분 뒤 까지의 예측이므로
    cost_MSE = MSE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])
    cost_MAPE = MAPE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])

    layer = tf.transpose(layer, perm=[1, 0])  # lstm에 unstack 이 있다면, 여기서는 transpose를 해주는 편이 위의 계산할 때 편할 듯
    Y = tf.transpose(Y, perm=[1, 0])  # y는 처음부터 잘 만들면 transpose할 필요 없지만, x랑 같은 batchslice를 하게 해주려면 이렇게 하는 편이 나음.

    # Pix2Pix
    DE = tf.concat([E[TIME_STAMP - 1], C[GEN_NUM - 1, :, 2, :, 0]], axis=1)


    loss_D = -tf.reduce_mean(
        tf.log(Discriminator_model(Y, DE, DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(
            1 - Discriminator_model(layer, DE, DISCRIMINATOR_BA, DISCRIMINATOR_DR, True)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(layer, DE, DISCRIMINATOR_BA,
                                                        DISCRIMINATOR_DR, True)))  + DISCRIMINATOR_ALPHA * train_MSE # MSE 는 0~ t까지 있어봤자 같은 값이다.
    '''
    loss_G_MSE = -tf.reduce_mean(tf.log(Discriminator_model(layer, DE, DISCRIMINATOR_BA,
                                                        DISCRIMINATOR_DR, True))) + DISCRIMINATOR_ALPHA * train_MSE
    '''
    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='discriminator_fc') #여기는 하나로 함수 합쳤음
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_fc') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_conv')  #다양해지면 여기가 모델마다 바뀜

    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator_fc')
    with tf.control_dependencies(D_update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_D, var_list=[vars_D, discriminator_weights]) #이 부분은 모델별로 고정

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_fc') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_conv')
    with tf.control_dependencies(G_update_ops):
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G, var_list=[vars_G ,fc_weights, conv_weights, convfc_weights])
    '''
    G_MSE_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_fc')
    with tf.control_dependencies(G_MSE_update_ops):
        train_G_MSE = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G_MSE, var_list=[vars_G, fc_weights])
    '''
    layer = tf.transpose(layer, perm=[1, 0])
    return train_MSE,cost_MAE, cost_MSE, cost_MAPE,layer[TIME_STAMP-1], train_D, train_G , loss_G#, train_G_MSE


#training 해준다.
def train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from):
    BATCH_NUM = int(len(train_idx) / BATCH_SIZE)
    print('BATCH_NUM: %d' % BATCH_NUM)
    min_mape = 100.0
    for _ in range(start_from):
        np.random.shuffle(train_idx)
    global_step_tr = 0
    global_step_te = 0
    for tr_idx in range(start_from, TRAIN_NUM):
        epoch_mse_cost = 0.0
        epoch_mape_cost = 0.0
        epoch_loss = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            #if LATENT_VECTOR_FLAG:
            C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', 12)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'ADV_FC')
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'ADV_FC')
            '''
            else:
                C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', 1)
                E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
                Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)
            '''
            if tr_idx > OPTIMIZED_EPOCH_CONV + PHASE1_EPOCH:
                _= sess.run([train_D], feed_dict={C:C_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
            if (tr_idx <= OPTIMIZED_EPOCH_CONV + PHASE1_EPOCH) | (tr_idx > OPTIMIZED_EPOCH_CONV + PHASE1_EPOCH + PHASE2_EPOCH):
                cost_MSE_val, cost_MAPE_val, cost_MSE_hist_val, _, loss= sess.run([cost_MSE, cost_MAPE, cost_MSE_hist, train_G, loss_G], feed_dict={C: C_train, E: E_train, Y: Y_train, BA: True,DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
                epoch_mse_cost += cost_MSE_val
                epoch_mape_cost += cost_MAPE_val
                epoch_loss += loss
                writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append([epoch_mse_cost / BATCH_NUM, epoch_mape_cost/BATCH_NUM])
            print("Train Cost %d: %lf %lf" % (tr_idx, epoch_mse_cost / BATCH_NUM, epoch_mape_cost/BATCH_NUM))
            print("Train loss %d: %lf" % (tr_idx, epoch_loss / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:
            if MASTER_SAVE_FLAG and (not ALL_TEST_SWITCH):
                sess.run(last_epoch.assign(tr_idx + 1))
                if (tr_idx) % SAVE_INTERVAL == 0:
                    print("Saving network...")
                    saver = tf.train.Saver()
                    if not os.path.exists(WHOLE_POINT_DIR):
                        os.makedirs(WHOLE_POINT_DIR)
                    saver.save(sess, WHOLE_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te = test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)
        # All test 해줌
        if ALL_TEST_SWITCH and test_result[tr_idx - OPTIMIZED_EPOCH_CONV - 1][2] < min_mape:
            if MASTER_SAVE_FLAG:
                sess.run(last_epoch.assign(tr_idx + 1))
                print("Saving network...")
                saver = tf.train.Saver()
                if not os.path.exists(WHOLE_POINT_DIR):
                    os.makedirs(WHOLE_POINT_DIR)
                saver.save(sess, WHOLE_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)
            print("alltest")
            min_mape = test_result[tr_idx - OPTIMIZED_EPOCH_CONV - 1][2]
            ALLTEST(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, np.array([i for i in range(0, 35350)]), sess, cr_idx, 'all')
        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다. GAN과는 상관없이 최종 MAE, MSE, MAPE만 뽑아준다.
def test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx))

    # Batch Slice
    C_test = batch_slice(C_data, test_idx, 0, 'CONV', 12, TEST_BATCH_SIZE)
    E_test = batch_slice(E_data, test_idx, 0, 'ADV_FC', 1, TEST_BATCH_SIZE)
    Y_test = batch_slice(Y_data, test_idx, 0, 'ADV_FC', 1, TEST_BATCH_SIZE)

    mae, mse, mape, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={C:C_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})


    writer_test.add_summary(cost_MAE_hist_val, global_step_te)
    writer_test.add_summary(cost_MSE_hist_val, global_step_te)
    writer_test.add_summary(cost_MAPE_hist_val, global_step_te)

    global_step_te += 1
    final_result[cr_idx].append(mape)
    test_result.append([mae , mse , mape ])
    print("Test Cost(%d) %d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, tr_idx, mae , mse , mape ))
    return global_step_te


def ALLTEST(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE,prediction, data_idx, sess, cr_idx, trainORtest):
    result_alltest = []

    file_name = 'ADV_CONV'

    for idx in range(len(data_idx)):
        C_test = batch_slice(C_data, data_idx, idx, 'CONV', 12, 1)
        E_test = batch_slice(E_data, data_idx, idx, 'ADV_FC', 1, 1)
        Y_test = batch_slice(Y_data, data_idx, idx, 'ADV_FC', 1, 1)
        mae, mse, mape,pred = sess.run([cost_MAE, cost_MSE, cost_MAPE,prediction], feed_dict={C:C_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})

        result_alltest.append([str(mae), str(mse), str(mape),str(pred[0])])


    if not os.path.exists(RESULT_DIR+'alltest/'):
        os.makedirs(RESULT_DIR+'alltest/')
    if FILEX_EXO.find("Zero") >= 0:
        resultfile = open(RESULT_DIR+'alltest/' + 'OnlySpeed_'+ file_name + '_alltest_'+ trainORtest +'_' + str(cr_idx) + '.csv', 'w', newline='')
    else:
        resultfile = open(RESULT_DIR+'alltest/' + 'Exogenous_' + file_name + '_alltest_' + trainORtest + '_' + str(cr_idx) + '.csv', 'w', newline='')
    output = csv.writer(resultfile)

    for idx in range(len(data_idx)):
        output.writerow(result_alltest[idx])

    resultfile.close()


###################################################-MAIN-###################################################
_, C_data, E_data, Y_data = input_data(0b011)
final_result = [[] for i in range(CROSS_ITERATION_NUM)]
_result_dir = RESULT_DIR + "CV" + str(CROSS_ITERATION_NUM) + "/" + "ADV_CONV"
cr_idx = 0
OS_OR_EXO = True
if FILEX_EXO.find("Zero") < 0:
    OS_OR_EXO = False
for train_idx, test_idx in load_Data():
    if cr_idx<3:
        cr_idx+=1
        continue
    print('CROSS VALIDATION: %d' % cr_idx)


    train_result = []
    test_result = []

    C = tf.placeholder("float32", [GEN_NUM, None, SPARTIAL_NUM, TEMPORAL_NUM, 1])
    E = tf.placeholder("float32", [GEN_NUM, None, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [GEN_NUM, None, 1])

    BA = tf.placeholder(tf.bool)
    DR = tf.placeholder(tf.float32)
    DISCRIMINATOR_BA = tf.placeholder(tf.bool)
    DISCRIMINATOR_DR = tf.placeholder(tf.float32)
    if RESTORE_GENERATOR_FLAG:
        last_epoch = tf.Variable(OPTIMIZED_EPOCH_CONV + 1, name=LAST_EPOCH_NAME)  # 받아올 방법이 없네..
    else:
        last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)
    init()
    sess = tf.Session()
    # 여기서는 모델만 외부 플래그, 그냥 train까지 외부 플래그 해도 됨
    train_MSE, cost_MAE, cost_MSE, cost_MAPE, prediction, train_D, train_G, loss_G = model_base(C, E, Y, BA, DR, DISCRIMINATOR_BA, DISCRIMINATOR_DR)
    if FILEX_EXO.find("Zero") >= 0:
        CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_OS_" + str(cr_idx) + "/"
        WHOLE_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_OS_WHOLE_" + str(cr_idx) + "/"
        writer_train = tf.summary.FileWriter("./tensorboard/adv_conv_os/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/adv_conv_os/test%d" % cr_idx, sess.graph)
    else:
        CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_EXO_" + str(cr_idx) + "/"
        WHOLE_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_EXO_WHOLE_" + str(cr_idx) + "/"
        writer_train = tf.summary.FileWriter("./tensorboard/adv_conv_exo/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/adv_conv_exo/test%d" % cr_idx, sess.graph)

    train_MSE_hist = tf.summary.scalar('train_MSE', train_MSE)
    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MAE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    if RESTORE_GENERATOR_FLAG:
        fc_batch_norm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope='generator_fc')
        conv_batch_norm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='generator_conv')
        variables_to_restore = fc_weights +conv_weights+convfc_weights+ fc_batch_norm_weights + conv_batch_norm_weights
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CURRENT_POINT_DIR)

    if RESTORE_FLAG:
        if checkpoint and checkpoint.model_checkpoint_path:
            # try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # except:
        #    print("Error on loading old network weights")
        else:
            print("Could not find old network weights")

    start_from = sess.run(last_epoch)
    # train my model
    print('Start learning from:', start_from)

    # train도 외부에서 FLAG해도됨. 지금은 안에 조건문이 있음
    train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE,prediction, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist,
          train_D, train_G, train_idx, test_idx, cr_idx, writer_train, writer_test, train_result, test_result,
          CURRENT_POINT_DIR, start_from)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'adv_conv'+ "_" + str(DISCRIMINATOR_LAYER_NUM) + "_" + str(LEARNING_RATE)[2:]+ "_" + format(DISCRIMINATOR_ALPHA,'f')[2:] + "_" + str(PHASE1_EPOCH) + "_"+ str(PHASE2_EPOCH) +"_"+ str(TRAIN_NUM)+ "_" , cr_idx, _result_dir)

    cr_idx = cr_idx + 1

    if (cr_idx == CROSS_ITERATION_NUM):
        break

output_result(final_result, 'adv_conv' + "_" + str(DISCRIMINATOR_LAYER_NUM) + "_" + str(LEARNING_RATE)[2:]+"_" + format(DISCRIMINATOR_ALPHA, 'f')[2:] + "_" + str(PHASE1_EPOCH) + "_"+ str(PHASE2_EPOCH)+"_"+ str(TRAIN_NUM)+ "_", cr_idx, _result_dir)