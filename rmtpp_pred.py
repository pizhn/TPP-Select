import tensorflow as tf
from tensorflow.contrib.keras import preprocessing
import tempfile
from model.non_linear.rmtpp import rmtpp_core
from func.constants import *
from predict.common import *

pad_sequences = preprocessing.sequence.pad_sequences


def rmtpp_data_process(timestamps, timestamp_dims, mark, pred_exo_order, train_test_ratio):
    train_rate, test_rate = train_test_ratio
    test_split_index = int(train_rate * len(timestamps))
    timestamps_train, timestamps_test = timestamps[:test_split_index], \
                                        timestamps[test_split_index:]
    mark_train, mark_test = mark[:test_split_index], mark[test_split_index:]
    timestamp_dims_train, timestamp_dims_test = timestamp_dims[:test_split_index], \
                                                timestamp_dims[test_split_index:]
    pred_exo_order_train, pred_exo_order_test = get_ranking(pred_exo_order[:test_split_index]), \
                                                get_ranking(pred_exo_order[test_split_index:])
    train_len, test_len = len(timestamps_train), len(timestamps_test)
    exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test = [None] * train_len, [None] * test_len
    timestamps_test -= timestamps_test[0]
    dim = np.unique(timestamp_dims).size
    eventTrain, eventTest = [list() for _ in range(dim)], [list() for _ in range(dim)]
    timeTrain, timeTest = [list() for _ in range(dim)], [list() for _ in range(dim)]

    for i in range(len(mark_train)):
        u = timestamp_dims_train[i]
        exo_order_cascade_idx_train[pred_exo_order_train[i]] = (u, len(eventTrain[u]))
        _mark = mark_train[i]
        _time = timestamps_train[i]
        u = timestamp_dims_train[i]
        eventTrain[u] += [_mark]
        timeTrain[u] += [_time]
    for i in range(len(mark_test)):
        u = timestamp_dims_test[i]
        exo_order_cascade_idx_test[pred_exo_order_test[i]] = (u, len(eventTest[u]))
        _mark = mark_test[i]
        _time = timestamps_test[i]
        u = timestamp_dims_test[i]
        eventTest[u] += [_mark]
        timeTest[u] += [_time]
    eventTrainIn = [x[:-1] for x in eventTrain]
    eventTrainOut = [x[1:] for x in eventTrain]
    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTrainIn = [x[:-1] for x in timeTrain]
    timeTrainOut = [x[1:] for x in timeTrain]
    timeTestIn = [x[:-1] for x in timeTest]
    timeTestOut = [x[1:] for x in timeTest]
    train_event_seq = pad_sequences(eventTrain, padding='post', dtype='float64')
    train_event_in_seq = pad_sequences(eventTrainIn, padding='post', dtype='float64')
    train_event_out_seq = pad_sequences(eventTrainOut, padding='post', dtype='float64')
    train_time_seq = pad_sequences(timeTrain, dtype=float, padding='post')
    train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
    train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')

    test_event_seq = pad_sequences(eventTest, padding='post', dtype='float64')
    test_event_in_seq = pad_sequences(eventTestIn, padding='post', dtype='float64')
    test_event_out_seq = pad_sequences(eventTestOut, padding='post', dtype='float64')
    test_time_seq = pad_sequences(timeTest, dtype=float, padding='post')
    test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
    test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding='post')

    data = {
        'train_event_in_seq': train_event_in_seq,
        'train_event_out_seq': train_event_out_seq,
        'train_time_in_seq': train_time_in_seq,
        'train_time_out_seq': train_time_out_seq,
        'test_event_in_seq': test_event_in_seq,
        'test_event_out_seq': test_event_out_seq,
        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,
        'num_categories': np.unique(mark).size
    }
    return data, train_event_seq, train_time_seq, test_event_seq, test_time_seq, \
           train_event_in_seq, train_event_out_seq, train_time_in_seq, train_time_out_seq, \
           test_event_in_seq, test_event_out_seq, test_time_in_seq, test_time_out_seq, \
           exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test, \
           train_len, test_len


def train(data, def_opts, endo_mask_out, epoches=30, bptt=5):
    tf.reset_default_graph()
    sess = tf.Session()
    model = rmtpp_core.RMTPP(
        sess=sess,
        num_categories=data['num_categories'],
        summary_dir=tempfile.mkdtemp(),
        batch_size=def_opts.batch_size,
        bptt=bptt,
        learning_rate=def_opts.learning_rate,
        cpu_only=True,
        _opts=def_opts
    )
    model.initialize(finalize=False)
    model.train_with_exo(endo_mask_out, data, restart=False,
                         with_summaries=False,
                         num_epochs=epoches, with_evals=False)
    return model


def accu(pred_times, true_times, pred_marks, true_marks):
    seq_limit = pred_times.shape[1]
    clipped_true_marks = true_marks[:, :seq_limit]
    valid_position = (true_times != 0)[:, :seq_limit]
    clipped_true_times = true_times[:, :seq_limit]
    solid_pred_mark = np.argmax(pred_marks, axis=-1)
    time_error = np.mean(np.abs(pred_times - clipped_true_times)[valid_position])
    mark_error = np.mean(solid_pred_mark[valid_position] != clipped_true_marks[valid_position])
    return time_error, mark_error


def get_endo_mask(cascade, cascade_exo_idx):
    endo_mask = np.full_like(cascade, True)
    for idx in cascade_exo_idx:
        endo_mask[idx[0]][idx[1]] = False
    return endo_mask


def rmtpp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark, k=1.):
    data, train_event_seq, train_time_seq, test_event_seq, test_time_seq, \
    train_event_in_seq, train_event_out_seq, train_time_in_seq, train_time_out_seq, \
    test_event_in_seq, test_event_out_seq, test_time_in_seq, test_time_out_seq, \
    exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test, \
    train_len, test_len = rmtpp_data_process(timestamps, timestamp_dims, mark, pred_exo_idxs, train_test)

    train_exo_num = int(train_len * k)
    train_cascade_exo_idx = [exo_order_cascade_idx_train[i] for i in range(train_exo_num)]
    train_cascade_exo_out_idx = [(i[0], i[1] - 1) for i in train_cascade_exo_idx if i[1] > 0]
    train_endo_mask_out = get_endo_mask(train_time_out_seq, train_cascade_exo_out_idx)
    model = train(data, rmtpp_core.def_opts, train_endo_mask_out)
    pred_times, pred_marks = model.predict(test_event_in_seq, test_time_in_seq)
    time_error, mark_error = accu(pred_times, test_time_out_seq, pred_marks, test_event_out_seq)
    return time_error, mark_error
