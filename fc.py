# -*- coding: utf-8 -*-
'''
----------------------------코드 설명----------------------------
-C-
1.FC에 해당하는 코드로
fc를 구현함
----------------------------고려 사항----------------------------


'''
from module import *
import os

#FC를 구현
def model(S, E, Y, BA, DR):
    layer = FC_model(S, E, BA, DR)
    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimal = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_MSE)

    return cost_MAE, cost_MSE, cost_MAPE, optimal, layer

#training 해준다.
def train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, optimal, train_idx, test_idx, cr_idx, writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from):
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
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            S_train = batch_slice(S_data, train_idx, ba_idx, 'FC', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)

            cost_MSE_val, cost_MAPE_val, cost_MSE_hist_val, _= sess.run([cost_MSE, cost_MAPE, cost_MSE_hist, optimal], feed_dict={S:S_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB})
            epoch_mse_cost += cost_MSE_val
            epoch_mape_cost += cost_MAPE_val
            writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append([epoch_mse_cost / BATCH_NUM, epoch_mape_cost / BATCH_NUM])
            print("Train Cost %d: %lf %lf" % (tr_idx, epoch_mse_cost / BATCH_NUM, epoch_mape_cost / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:
            sess.run(last_epoch.assign(tr_idx + 1))
            if tr_idx % SAVE_INTERVAL == 0:
                if not ALL_TEST_SWITCH:
                    print("Saving network...")
                    if not os.path.exists(CURRENT_POINT_DIR):
                        os.makedirs(CURRENT_POINT_DIR)
                    saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)
                if tr_idx == OPTIMIZED_EPOCH_FC:
                    print("Saving network for ADV...")
                    if not os.path.exists(ADV_POINT_DIR):
                        os.makedirs(ADV_POINT_DIR)
                    saver.save(sess, ADV_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)
            global_step_te=test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)

        # All test 해줌
        if ALL_TEST_SWITCH and test_result[tr_idx][2]<min_mape:
            sess.run(last_epoch.assign(tr_idx + 1))
            print("Saving network...")
            if not os.path.exists(CURRENT_POINT_DIR):
                os.makedirs(CURRENT_POINT_DIR)
            saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)
            print("alltest")
            min_mape = test_result[tr_idx][2]
            ALLTEST(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, np.array([i for i in range(0,35350)]), sess, cr_idx, 'all')

        #cross validation의 train_idx를 shuffle해준다.
        print("%d" % train_idx[0])
        np.random.shuffle(train_idx)



#testing 해준다.
def test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx))
    # Batch Slice
    S_test = batch_slice(S_data, test_idx, 0, 'FC', 1, TEST_BATCH_SIZE)
    E_test = batch_slice(E_data, test_idx, 0, 'FC', 1, TEST_BATCH_SIZE)
    Y_test = batch_slice(Y_data, test_idx, 0, 'FC', 1, TEST_BATCH_SIZE)

    mae, mse, mape, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={S:S_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB})

    writer_test.add_summary(cost_MAE_hist_val, global_step_te)
    writer_test.add_summary(cost_MSE_hist_val, global_step_te)
    writer_test.add_summary(cost_MAPE_hist_val, global_step_te)

    global_step_te+=1

    test_result.append([mae, mse , mape])
    final_result[cr_idx].append(mape)
    print("Test Cost(%d) %d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, tr_idx, mae , mse , mape ))

    return global_step_te

#batch slice부분 모델마다 다르게 해줘야함.(각 모델의test참고)
def ALLTEST(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, data_idx, sess, cr_idx, trainORtest):
    result_alltest = []

    file_name = 'FC'

    for idx in range(len(data_idx)):
        S_test = batch_slice(S_data, data_idx, idx, file_name, 1, 1)
        E_test = batch_slice(E_data, data_idx, idx, file_name, 1, 1)
        Y_test = batch_slice(Y_data, data_idx, idx, file_name, 1, 1)
        mae, mse, mape, pred = sess.run([cost_MAE, cost_MSE, cost_MAPE, prediction], feed_dict={S:S_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB})

        result_alltest.append([str(mae), str(mse), str(mape), str(pred[0][0])])


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
S_data, _, E_data, Y_data = input_data(0b101) #speed, exogenous 사용
final_result = [[] for i in range(CROSS_ITERATION_NUM)]
_result_dir = RESULT_DIR + "CV" + str(CROSS_ITERATION_NUM) + "/" + "FC"
DIR = "./index/"
if not os.path.exists(DIR):
    os.makedirs(DIR)
OS_OR_EXO = True
if FILEX_EXO.find("Zero") < 0:
    OS_OR_EXO = False
cr_idx =0
for train_idx, test_idx in load_Data():


    train_result = []
    test_result = []

    S = tf.placeholder("float32", [None, TIME_STAMP])
    E = tf.placeholder("float32", [None, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [None, 1])
    BA = tf.placeholder(tf.bool)
    DR = tf.placeholder(tf.float32)
    last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

    sess = tf.Session()

    init()
    cost_MAE, cost_MSE, cost_MAPE, optimal, prediction = model(S, E, Y, BA, DR)

    if FILEX_EXO.find("Zero") >= 0:
        CURRENT_POINT_DIR = CHECK_POINT_DIR + "FC_OS_" + str(cr_idx) + "/"
        ADV_POINT_DIR = CHECK_POINT_DIR + "ADV_FC_OS_" + str(cr_idx) + "/"
        writer_train = tf.summary.FileWriter("./tensorboard/fc_os/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/fc_os/test%d" % cr_idx, sess.graph)
    else:
        CURRENT_POINT_DIR = CHECK_POINT_DIR + "FC_EXO_" + str(cr_idx) + "/"
        ADV_POINT_DIR = CHECK_POINT_DIR + "ADV_FC_EXO_" + str(cr_idx) + "/"
        writer_train = tf.summary.FileWriter("./tensorboard/fc_exo/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/fc_exo/test%d" % cr_idx, sess.graph)

    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CURRENT_POINT_DIR)

    if RESTORE_FLAG:
        if checkpoint and checkpoint.model_checkpoint_path:
            #try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            #except:
            #    print("Error on loading old network weights")
        else:
            print("Could not find old network weights")

    start_from = sess.run(last_epoch)
    # train my model
    print('Start learning from:', start_from)

    train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, prediction, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, optimal, train_idx, test_idx, cr_idx, writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'fc_'+str(FC_LAYER_NUM)+'layer_', cr_idx, _result_dir)
    cr_idx = cr_idx + 1
    if (cr_idx == CROSS_ITERATION_NUM):
        break

output_result(final_result, 'fc' + "_", cr_idx , _result_dir)