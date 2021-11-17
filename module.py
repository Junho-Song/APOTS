# -*- coding: utf-8 -*-
'''
----------------------------코드 설명----------------------------
-C-
module 파일: 헤더 파일
구성 요소
각종 hyper parameter

----------------------------고려 사항----------------------------
*2019 07 05
upStream_num, downStream_num은 데이터를 만들때 고려해서 실험 해줘야함
그래서 mu, md 라는 변수를 만들긴 하지만 나중에 파일에서 읽어와야함
*2019 07 20
과연 lstm을 미리 preprocess하는것이 효과가 있을까 그냥 fully data에서 받아와서 사용하는것이 더 효율적이지 않을까
input_data에서 안사용하는 데이터도 모두 받아와야하는데 이를 조건문으로 바꿔줄 필요가있다.
*2019 07 21
conv의 filter size와 layer등의 default값을 정해주어야 한다.
각각의 module이 batch를 제대로 반영하고 있는지 확인해야함
*2019 07 22
66 -> 80? 바꿔야합니다
tf.reduce_mean 이 이 실험에서 무슨 의미인가?

Q. batch slice에서 +를 통해 data index에 접근하는데 이때 최대치를 넘어버릴 수 있다
A. Cross validation 할때 Cell size만큼을 빼고 train_idx와 test_idx를 구해준다.

*2019 07 23
Q. conv에 뒤에 32채널을 1채널로 바꾸고 12개의 속도 데이터로 concat해주는 코드가 작성 되어야 한다.
A. 해결했다

Q. SPEED_MAX와 SPEED_MIN값 찾아야한다

Q. conv 채널수 늘리기 정확도가 fc랑 비슷하면 안된다.

'''

# import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import csv
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(777)  # KFold 의 shuffle과 batch shuffle의 seed를 설정 해준다
tf.set_random_seed(777)  # tf.random의 seed 설정
random.seed(777)

# Setting


# File name
FILEX_SPEED = '../Data/Speed/x_Base_data_2016204_5min_60min_60min_only_speed.csv'  # speed만 잘라낸 파일 이름(X data)
FILEX_EXO = '../Data/Exogenous/ExogenousBase_data_2016204_5min_60min_60min_8.csv'  # exogenous(data 8)만 잘라낸 파일 이름(X data)
FILEX_CONV = '../Data/Convolution/x_data_2016204_5min_60min_60min_only_speed.csv'  # preprocessing한 conv data 파일 이름(X data)
FILEY = '../Data/Y/y_data_2016204_5min_60min_60min.csv'  # beta분 후 speed 파일 이름(Y data)
CHECK_POINT_DIR = './save/'  # 각 weight save 파일의 경로입니다.
RESULT_DIR = './Result/'
LAST_EPOCH_NAME = 'last_epoch'  # 불러온 에폭에 대한 이름입니다.
OPTIMIZED_EPOCH_FC = 35  # SAVE_INTERVEL 의 배수여야 합니다.
OPTIMIZED_EPOCH_CONV = 30  # SAVE_INTERVEL 의 배수여야 합니다.
OPTIMIZED_EPOCH_LSTM = 30  # SAVE_INTERVEL 의 배수여야 합니다.
OPTIMIZED_EPOCH_CONV_LSTM = [15, 10, 20, 10, 10]  # SAVE_INTERVEL 의 배수여야 합니다.15
PHASE1_EPOCH = 10
PHASE2_EPOCH = 20

ALL_TEST_SWITCH = False  # Alltest  on/off

# FLAG
# USE_LOAD = True
RESTORE_FLAG = False  # weight 불러오기 여부 [default False]  두개 나눠져야함. 전체 불러올때는 restore만 true로 generator는 false로
RESTORE_GENERATOR_FLAG = False  # Generator weight 불러오기 여부 [RESTORE_FLAG]가 False 이면 항상 False[default False] 한번 저장된 후에는 False로 무조건 바꿔야 합니다.
MASTER_SAVE_FLAG = False  # [WARNING] 저장이 되지 않습니다. (adv 모델에 한해 적용)

# Fix value(Week Cross Validation)
DAY = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
FIRST_MONTH = 7
LAST_MONTH = 10
ONE_DAY = 288
ONE_WEEK = ONE_DAY * 7
WEEK_NUM = 4
INTERVAL = 24  # adv conv lstm에서 overlap방지

# variable
TRAIN_NUM = 21  # traing 회수 [default 1000]
SPEED_MAX = 98  # data내의 최고 속도 [default 100]
SPEED_MIN = 3  # data내의 최저 속도 [default 0]
CROSS_NUM = 5  # cross validation의 spilit 수
CROSS_ITERATION_NUM = 5  # cross validation의 반복수 (CROSS_NUM보다 작아야하며 독립적으로 생각됨)
BATCH_SIZE = 300  # 1 epoch 당 batch의 개수 [default 300]
TEST_BATCH_SIZE = 147
LEARNING_RATE = 0.001  # learning rate(모든 model, gan은 *2)
TRAIN_PRINT_INTERVAL = 1  # train 에서 mse값 출력 간격
TEST_PRINT_INTERVAL = 1  # test 에서 mae, mse, mape값 출력 간격
SAVE_INTERVAL = 5

# Hyper Parameter(FC)
FC_LAYER_NUM = 5  # fc layer의 깊이 [default 3]
VECTOR_SIZE = 95  # fc와 lstm에 들어가는 vector의 크기 [default 83]
TIME_STAMP = 12  # lstm과 fc의 vector에서 고려해주는 시간 [default 12]
EXOGENOUS_NUM = VECTOR_SIZE - TIME_STAMP  # exogenous로 들어가는 data의 개수 [default 73]
LAYER_UNIT_NUM = [VECTOR_SIZE, 512, 128, 256, 64, 1]  # fc에서 고려해줄 layer당 unit의 수 default[83, 64, 128, 64, 1]
FC_BATCH_NORM = True  # fc 에서 batch normalization 을 사용할것인지 [default True]
FC_DROPOUT = True  # fc 에서 drop out 을 사용할것인지 [default True]
FC_TR_KEEP_PROB = 0.8  # training 에서 dropout 비율
FC_TE_KEEP_PROB = 1.0  # testing 에서 dropout 비율

# Hyper Parameter(CONV)
POOLING = False  # pooling을 사용할 것인지 [default True]
CONV_BATCH_NORM = True  # conv 에서 batch normalization 을 사용할것인지 [default True]
CONV_LAYER_NUM = 3  # conv layer의 깊이 [default 3]
TEMPORAL_NUM = 12  # conv에서 고려할 시간 default 12]
UP_STREAM_NUM = 2  # conv에서 고려할 이후의 도로 개수들 [default 2]
DOWN_STREAM_NUM = 2  # conv에서 고려할 이전의 도로 개수들 [default 2]
SPARTIAL_NUM = DOWN_STREAM_NUM + UP_STREAM_NUM + 1  # conv에서 고려할 총 도로의 수 + 타겟도로[default 13]
CHANNEL_NUM = [1, 128, 32,
               64]  # conv에서 고려해줄 channel 수 [default 1 64 16 32] **주의 1로 시작해서 1로 끝나야함 input과 ouput channel은 1개씩이기 때문
FILTER_SIZE_TEMPORAL = [3, 1, 3]  # 시간의 filter size [default 3 1 3]
FILTER_SIZE_SPATIAL = [3, 1, 3]  # 공간의 filter size [default 3 1 3]
LAST_LAYER_SIZE = 8

# Hyper Parameter(LSTM)
HIDDEN_NUM = [512, 512]  # lstm의 hidden unit 수 [default 32]
FORGET_BIAS = 1.0  # lstm의 forget bias [default 1.0]
CELL_SIZE = 12  # lstm의 cell 개수 [default 12]
GEN_NUM = 12  # generator의 개수

# Hyper Parameter(Discriminator)
DISCRIMINATOR_INPUT_NUM = 107  # discriminator conv 이면 83 FC 이면 84
DISCRIMINATOR_LAYER_NUM = 5
DISCRIMINATOR_LAYER_UNIT_NUM = [DISCRIMINATOR_INPUT_NUM, 512, 128, 256, 64, 1]
DISCRIMINATOR_BATCH_NORM = True
DISCRIMINATOR_DROPOUT = True
DISCRIMINATOR_TR_KEEP_PROB = 0.8  # training 에서 dropout 비율
DISCRIMINATOR_TE_KEEP_PROB = 1.0  # testing 에서 dropout 비율
DISCRIMINATOR_ALPHA = 0.00008  # MSE 앞에 붙는 람다 term

# Hyper Parameter(PEEK_DATA)
TEST_CASE_NUM = 20
TEST_RATIO = 10
TIME_INTERVAL = 24
DATA_SIZE = 35400 - TIME_STAMP

fc_weights = []  # fc weight들의 크기는 layer의 길이에 따라 결정된다.
discriminator_weights = []  # 여기서 부터는 E가 들어가는 지점
discriminator_conv_weights = []
discriminator_convfc_weights = []
conv_weights = []  # conv weight들의 크기는 layer의 길이에 따라 결정된다.
convfc_weights = []  # conv 이후 fc의 weight
lstm_weights = []  # lstm weight들의 크기는 layer의 길이에 따라 결정된다.
lstm_biases = []  # lstm bias들의 크기는 layer의 길이에 따라 결정된다.


# weight를 만들어준다.
def init():
    # 모든 weight를 clear 해준다.
    fc_weights.clear()
    conv_weights.clear()
    convfc_weights.clear()
    lstm_weights.clear()
    lstm_biases.clear()
    discriminator_weights.clear()
    discriminator_conv_weights.clear()

    # fc weight 초기화
    for layer_idx in range(1, FC_LAYER_NUM + 1):
        fc_weights.append(init_weights([LAYER_UNIT_NUM[layer_idx - 1], LAYER_UNIT_NUM[layer_idx]]))

    # conv weight 초기화
    for layer_idx in range(1, CONV_LAYER_NUM + 1):
        conv_weights.append(init_weights(
            [FILTER_SIZE_SPATIAL[layer_idx - 1], FILTER_SIZE_TEMPORAL[layer_idx - 1], CHANNEL_NUM[layer_idx - 1],
             CHANNEL_NUM[layer_idx]]))
    convfc_weights.append(init_weights([LAST_LAYER_SIZE * CHANNEL_NUM[CONV_LAYER_NUM], TIME_STAMP]))

    # lstm weight 초기화
    lstm_weights.append(init_weights([HIDDEN_NUM[-1], 1]))

    lstm_biases.append(init_weights([1]))

    # discriminator weight 초기화
    for layer_idx in range(1, DISCRIMINATOR_LAYER_NUM + 1):
        discriminator_weights.append(
            init_weights([DISCRIMINATOR_LAYER_UNIT_NUM[layer_idx - 1], DISCRIMINATOR_LAYER_UNIT_NUM[layer_idx]]))


# shper를 input으로 받아 weight를 initailization 해줌
def init_weights(input_shape):
    return tf.Variable(tf.random_normal(input_shape, stddev=0.01))  # name은 임의로 정했음

# file을 numpy array 데이터로 받아옴
def fileToData(fileName):
    data_list = []
    File = open(fileName, 'r')
    FileData = csv.reader(File)
    for line in FileData:
        temp_line = []
        for line_idx in range(len(line)):
            temp_line.append(float(line[line_idx]))
        data_list.append(temp_line)
    File.close()
    return np.array(data_list)


# input data를 만들어줌
# type 2진법 ex) 0b111 4자리는 Speed, 2자리는 Conv, 1자리는 exogenous
def input_data(type):
    S_data = np.array([])
    C_data = np.array([])
    E_data = np.array([])
    # file을 numpy로 바꿔줌
    # &로 각 자리를 비교해준다.
    if type & 0b100 != False:
        S_data = fileToData(FILEX_SPEED)  # only speed 데이터(시간순)
    if type & 0b10 != False:
        C_data = fileToData(FILEX_CONV)  # conv 데이터(행(공간), 열(시간))
    if type & 0b1 != False:
        E_data = fileToData(FILEX_EXO)  # 외부요소만 자른 데이터
    Y_data = fileToData(FILEY)  # 실재값 데이터
    return S_data, C_data, E_data, Y_data


# 에러계산식
def MAE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.abs(y_test_orig - y_pred_orig))


def MSE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.square((y_test_orig - y_pred_orig)))


def MAPE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100


# FC_model로 input으로 CNN output이 output으로 예측 속도값이 나온다.
def FC_model(S, E, BA, DR, is_reuse=False):
    with tf.variable_scope('generator_fc', reuse=is_reuse):
        batch_prob = BA
        dropout_prob = DR

        for layer_idx in range(FC_LAYER_NUM):
            if layer_idx != 0:
                layer = tf.matmul(layer, fc_weights[layer_idx])
            else:
                layer = tf.matmul(tf.concat([S, E], axis=1), fc_weights[layer_idx])
            if FC_BATCH_NORM == True:
                layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=batch_prob)
            layer = tf.nn.relu(layer)

            # if FC_DROPOUT == True:
            # layer = tf.nn.dropout(layer, keep_prob=dropout_prob) #조사할 필요가 있음

    return layer


# CONV network로 input으로 시공간 입력이 output으로 layer가 나온다 여기서는 X 가 메트릭스
def CNN_model(X, BA, is_reuse=False):
    with tf.variable_scope('generator_conv', reuse=is_reuse):
        batch_prob = BA

        for layer_idx in range(CONV_LAYER_NUM):
            if layer_idx != 0:
                layer = tf.nn.conv2d(layer, conv_weights[layer_idx], strides=[1, 1, 1, 1], padding='VALID')
            else:
                layer = tf.nn.conv2d(X, conv_weights[layer_idx], strides=[1, 1, 1, 1], padding='VALID')

            if CONV_BATCH_NORM == True:
                layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=batch_prob)
            layer = tf.nn.relu(layer)
            if POOLING == True and layer_idx != (CONV_LAYER_NUM - 1):  # 마지막 layer는 pooling안함
                layer = tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1])

        layer = tf.reshape(layer, shape=[-1, CHANNEL_NUM[CONV_LAYER_NUM] * LAST_LAYER_SIZE])
        layer = tf.matmul(layer, convfc_weights[0])
        layer = tf.nn.relu(layer)

        # **fc 하나 추가해 주어야함

    return layer


# LSTM network로 input으로 X(batch_size * speed_size * cell_size)와 E(batch_size * exogenous_size * cell_size)가 들어온다.
# output으로 마지막 예측값만 내놓는다.
# 현재: time stamp 12, vector_size 66, cell_size 12, output 1
# 추후에 실험 1,2 해봐야함
# 실험1: time stamp 1, vector_size 6?7?, cell_size 12, output 1
# 실험2: time stamp 12, vector_size 66, cell_size 12, output 12
def LSTM_model(S, E, is_reuse=False):
    with tf.variable_scope('generator_lstm', reuse=is_reuse):
        # 66(vector_size) * 12(cell size)를 나눠줌
        # X,E는 같은 시간 끼리 합쳐줌
        x = tf.unstack(tf.concat([S, E], axis=2), axis=0)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM[0], forget_bias=FORGET_BIAS)
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias=FORGET_BIAS)

        outputs, _ = tf.nn.static_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
        # outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], lstm_weights[0]) + lstm_biases[0]


def multi_LSTM_model(S, E, isReuse=False):
    with tf.variable_scope('generator_lstm', reuse=isReuse):
        x = tf.unstack(tf.concat([S, E], axis=2), axis=0)

        cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in HIDDEN_NUM]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # lstm_cell1 = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM, forget_bias=FORGET_BIAS)
        # lstm_cell2  = tf.nn.rnn_cell.LSTMCell(num_units=256, forget_bias=FORGET_BIAS)
        # multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2])

        outputs, _ = tf.nn.static_rnn(cell=multi_cell, inputs=x, dtype=tf.float32)
        # outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], lstm_weights[0]) + lstm_biases[0]


def LSTM_model_12(S, E, is_reuse=False):
    with tf.variable_scope('generator_lstm', reuse=is_reuse):  # 원래 lstm과 weight share
        # 66(vector_size) * 12(cell size)를 나눠줌
        # X,E는 같은 시간 끼리 합쳐줌
        x = tf.unstack(tf.concat([S, E], axis=2), axis=0)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM, forget_bias=FORGET_BIAS)
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias=FORGET_BIAS)

        outputs, _ = tf.nn.static_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
        # outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, lstm_weights) + lstm_biases


def multi_LSTM_model_12(S, E, isReuse=False):
    with tf.variable_scope('generator_lstm', reuse=isReuse):
        x = tf.unstack(tf.concat([S, E], axis=2), axis=0)

        cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in HIDDEN_NUM]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # lstm_cell1 = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM, forget_bias=FORGET_BIAS)
        # lstm_cell2  = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM, forget_bias=FORGET_BIAS)
        # multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2])

        outputs, _ = tf.nn.static_rnn(cell=multi_cell, inputs=x, dtype=tf.float32)
        # outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, lstm_weights) + lstm_biases


# discriminator 의 X는 y 와 predicted y 가 concatenated 되어서 들어온 13짜리 X입니다. 기존의 S랑 다름 -> 매우 중요, 또는 conv일떈 12짜리 예측 벡터
def Discriminator_model(X, E, DISCRIMINATOR_BA, DISCRIMINATOR_DR, is_reuse=False):
    with tf.variable_scope('discriminator_fc', reuse=is_reuse):
        discriminator_batch_prob = DISCRIMINATOR_BA
        discriminator_dropout_prob = DISCRIMINATOR_DR
        for layer_idx in range(DISCRIMINATOR_LAYER_NUM):  # same as FC_LAYER_NUM
            if layer_idx != 0:
                layer = tf.matmul(layer, discriminator_weights[layer_idx])
            else:
                layer = tf.matmul(tf.concat([X, E], axis=1), discriminator_weights[layer_idx])

            if DISCRIMINATOR_BATCH_NORM == True:
                layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=discriminator_batch_prob)
            # 마지막 레이어는 Sigmoid logistic regression, 마지막 출력이 1이라는 가정 하에 작성합니다
            if layer_idx == DISCRIMINATOR_LAYER_NUM - 1:
                layer = tf.nn.sigmoid(layer)
            else:
                layer = tf.nn.relu(layer)

            # if DISCRIMINATOR_DROPOUT == True:
            # layer = tf.nn.dropout(layer, keep_prob=discriminator_dropout_prob)
    return layer


# type에 따라 다른 batch slice 결과를 내어준다.
# da_idx는 cross validation해서 나온 idx의 집합
# ba_idx는 batch의 idx
# cell size는 conv+lstm에서 고려해줘야할 conv의 수
def batch_slice(data, data_idx, batch_idx, slice_type, cell_size=1, BATCH_SIZE=300):
    # fc X input data와 fc, conv의 y output data
    if slice_type == 'FC':
        slice_data = data[data_idx[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]]

    elif slice_type == 'ADV_FC':
        for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
            start_idx = data_idx[idx]
            if idx == batch_idx * BATCH_SIZE:
                slice_data = data[start_idx: start_idx + TIME_STAMP].reshape(TIME_STAMP, 1,
                                                                             -1)  # 마지막이 -1인 이유(speed의 경우 12 이고 exogenous의 경우 71이기 때문)
            else:
                slice_data = np.append(slice_data, data[start_idx: start_idx + TIME_STAMP].reshape(TIME_STAMP, 1, -1),
                                       axis=1)  # [TIME_STAMP, BATCH_SIZE, -1]  LSTM의  CELL STATE 와 다름


    # conv X input data, cell size에 따라 연속된 conv input을 뽑을수있다.(lstm의 input으로 들어가기 위한)
    elif slice_type == 'CONV':
        for cell_idx in range(cell_size):
            for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
                start_idx = data_idx[idx]
                if idx == batch_idx * BATCH_SIZE:
                    temp = data[
                           (start_idx + cell_idx) * SPARTIAL_NUM: ((start_idx + cell_idx + 1) * SPARTIAL_NUM)].reshape(
                        1, 1, SPARTIAL_NUM, TEMPORAL_NUM, 1)
                else:
                    temp = np.append(temp, data[(start_idx + cell_idx) * SPARTIAL_NUM: (
                            (start_idx + cell_idx + 1) * SPARTIAL_NUM)].reshape(1, 1, SPARTIAL_NUM, TEMPORAL_NUM,
                                                                                1), axis=1)

            if cell_idx == 0:
                slice_data = temp
            else:
                slice_data = np.append(slice_data, temp, axis=0)
    # slice data 형태 [GEN_NUM, CELL_SIZE, BATCH_SIZE, SPATIAL, TEMPORAL, CHANNEL]
    elif slice_type == 'ADV_CONV':
        for gen_idx in range(GEN_NUM):
            for cell_idx in range(cell_size):
                for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
                    start_idx = data_idx[idx]
                    if idx == batch_idx * BATCH_SIZE:
                        temp_batch = data[(start_idx + cell_idx + gen_idx) * SPARTIAL_NUM: (
                                (start_idx + cell_idx + gen_idx + 1) * SPARTIAL_NUM)].reshape(1, 1, 1, SPARTIAL_NUM,
                                                                                              TEMPORAL_NUM, 1)
                    else:
                        temp_batch = np.append(temp_batch, data[(start_idx + cell_idx + gen_idx) * SPARTIAL_NUM: (
                                (start_idx + cell_idx + gen_idx + 1) * SPARTIAL_NUM)].reshape(1, 1, 1, SPARTIAL_NUM,
                                                                                              TEMPORAL_NUM, 1),
                                               axis=2)

                if cell_idx == 0:
                    temp_cell = temp_batch
                else:
                    temp_cell = np.append(temp_cell, temp_batch, axis=1)

            if gen_idx == 0:
                slice_data = temp_cell
            else:
                slice_data = np.append(slice_data, temp_cell, axis=0)

    # lstm X input data
    elif slice_type == 'LSTM':
        for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
            start_idx = data_idx[idx]
            if idx == batch_idx * BATCH_SIZE:
                slice_data = data[start_idx: start_idx + CELL_SIZE].reshape(CELL_SIZE, 1,
                                                                            -1)  # 마지막이 -1인 이유(speed의 경우 12 이고 exogenous의 경우 71이기 때문)
            else:
                slice_data = np.append(slice_data, data[start_idx: start_idx + CELL_SIZE].reshape(CELL_SIZE, 1, -1),
                                       axis=1)
    # slice data 형태 [GEN_NUM, CELL_SIZE, BATCH_SIZE, VECTOR]
    elif slice_type == 'ADV_LSTM':
        for gen_idx in range(GEN_NUM):
            for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
                start_idx = data_idx[idx]
                if idx == batch_idx * BATCH_SIZE:
                    temp_batch = data[start_idx + gen_idx: start_idx + gen_idx + CELL_SIZE].reshape(1, CELL_SIZE, 1,
                                                                                                    -1)  # 마지막이 -1인 이유(speed의 경우 12 이고 exogenous의 경우 71이기 때문)
                else:
                    temp_batch = np.append(temp_batch,
                                           data[start_idx + gen_idx: start_idx + gen_idx + CELL_SIZE].reshape(1,
                                                                                                              CELL_SIZE,
                                                                                                              1, -1),
                                           axis=2)
            if gen_idx == 0:
                slice_data = temp_batch
            else:
                slice_data = np.append(slice_data, temp_batch, axis=0)

    # lstm의 output data(60분 후를 뽑아야 하기때문)
    elif slice_type == 'LSTMY':
        slice_data = data[data_idx[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE] + (CELL_SIZE - 1)]


    elif slice_type == 'ADV_LSTMY':
        for gen_idx in range(GEN_NUM):
            temp_gen = data[
                data_idx[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE] + CELL_SIZE - 1 + gen_idx].reshape(
                BATCH_SIZE, 1)
            if gen_idx == 0:
                slice_data = temp_gen
            else:
                slice_data = np.append(slice_data, temp_gen, axis=1)


    else:
        print('ERROR: slice type error\n')

    return slice_data


# train과 test에서 얻은 결과를 file로 만든다.
# file_name에 실행하는 코드의 이름을 적는다 ex)adv_conv_lstm
def output_data(train_result, test_result, file_name, cr_idx, _result_dir):
    # train output
    if not os.path.exists(_result_dir + "_OS/"):
        os.makedirs(_result_dir + "_OS/")
    if not os.path.exists(_result_dir + "_EXO/"):
        os.makedirs(_result_dir + "_EXO/")
    if FILEX_EXO.find("Zero") >= 0:
        outputfile = open(_result_dir + "_OS/" + 'OnlySpeed_' + file_name + str(cr_idx) + '_tr' + '.csv', 'w',
                          newline='')
    else:
        outputfile = open(_result_dir + "_EXO/" + 'Exogenous_' + file_name + str(cr_idx) + '_tr' + '.csv', 'w',
                          newline='')

    output = csv.writer(outputfile)

    for tr_idx in range(len(train_result)):
        output.writerow([str(train_result[tr_idx][0]), str(train_result[tr_idx][1])])

    outputfile.close()

    # test output
    if FILEX_EXO.find("Zero") >= 0:
        outputfile = open(_result_dir + "_OS/" + 'OnlySpeed_' + file_name + str(cr_idx) + '_te' + '.csv', 'w',
                          newline='')
    else:
        outputfile = open(_result_dir + "_EXO/" + 'Exogenous_' + file_name + str(cr_idx) + '_te' + '.csv', 'w',
                          newline='')

    output = csv.writer(outputfile)
    for te_idx in range(len(test_result)):
        output.writerow([str(test_result[te_idx][0]), str(test_result[te_idx][1]), str(test_result[te_idx][2])])

    outputfile.close()


def output_result(final_result, file_name, cr_idx, _result_dir):
    ''' REDUNDENT
    if not os.path.exists(_result_dir + "_OS/"):
        os.makedirs(_result_dir + "_OS/")
    if not os.path.exists(_result_dir + "_EXO/"):
        os.makedirs(_result_dir + "_EXO/")
    '''
    if FILEX_EXO.find("Zero") >= 0:
        resultfile = open(
            _result_dir + "_OS/" + 'OnlySpeed_' + file_name + 'result' + '_' + str(CROSS_ITERATION_NUM) + '.csv', 'w',
            newline='')
    else:
        resultfile = open(
            _result_dir + "_EXO/" + 'Exogenous_' + file_name + 'result' + '_' + str(CROSS_ITERATION_NUM) + '.csv', 'w',
            newline='')
    output = csv.writer(resultfile)

    if cr_idx == CROSS_ITERATION_NUM:
        total_result = []
        for te_idx in range(len(final_result[0])):
            mean_result = 0.0
            row = []
            for cr_idx in range(CROSS_ITERATION_NUM):
                mean_result += final_result[cr_idx][te_idx]
                row.append(str(final_result[cr_idx][te_idx]))
            mean_result /= CROSS_ITERATION_NUM
            total_result.append(mean_result)
            row.append(str(mean_result))
            output.writerow(row)
        output.writerow(
            ['epoch(Excel):', str(total_result.index(min(total_result)) + 1), 'min_value:', str(min(total_result))])
    else:
        print("cannot save result, cross num and iteration num must be 20")

    resultfile.close()


def load_Data():
    train_idx = [[] for _ in range(TEST_CASE_NUM)]
    test_idx = [[] for _ in range(TEST_CASE_NUM)]

    DIR = "./index/"

    for i in range(TEST_CASE_NUM):
        File = open(DIR + "te" + str(i) + ".csv", 'r')
        FileData = csv.reader(File)
        for line in FileData:
            test_idx[i].append(int(line[0]))
        File.close()
        test_idx[i] = np.array(test_idx[i], dtype="int64")

        File = open(DIR + "tr" + str(i) + ".csv", 'r')
        FileData = csv.reader(File)
        for line in FileData:
            train_idx[i].append(int(line[0]))
        File.close()
        train_idx[i] = np.array(train_idx[i], dtype="int64")

    return zip(np.array(train_idx), np.array(test_idx))
