import argparse, torch
import numpy as np
from torch import autograd
from func.constants import *
from predict.common import *

from model.non_linear.sahp.train_functions.train_sahp import train_eval_sahp_exo, prediction
import torch.nn as nn

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_LEARN_RATE = 5e-5

device = 'cpu'
USE_CUDA = False
args.atten_heads = 8
batch_size = 32
bias = False
d_model = 16
dropout = 0.1
hidden_size = 16
lambda_l2 = 3e-4
lr = 5e-5
nLayers = 4
EPOCHS = 20


def process_list(event_times_list, event_types_list, dim):
    tmax = 0
    for tsr in event_times_list:
        if tsr.size(0) != 0 and torch.max(tsr) > tmax:
            tmax = torch.max(tsr)

    seq_lengths = torch.tensor([t.size(0) for t in event_times_list])
    #  Build a data tensor by padding
    seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value=tmax).float()
    seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim=1)  # add 0 to the sequence beginning

    seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value=dim)
    seq_types = torch.cat(
        (dim * torch.ones_like(seq_types[:, :1]), seq_types), dim=1).long()  # convert from floattensor to longtensor
    return seq_times, seq_types, seq_lengths, tmax


def sahp_data_preprocess(timestamps, timestamp_dims, mark, pred_exo_idxs):
    train_rate, valid_rate, test_rate = train_valid_test
    valid_split_index = int(train_rate * len(timestamps))
    test_split_index = int((train_rate + valid_rate) * len(timestamps))
    timestamps_train, timestamps_valid, timestamps_test = timestamps[:valid_split_index], \
                                                          timestamps[valid_split_index:test_split_index], \
                                                          timestamps[test_split_index:]
    mark_train, mark_valid, mark_test = mark[:valid_split_index], mark[valid_split_index:test_split_index], mark[
                                                                                                            test_split_index:]
    timestamp_dims_train, timestamp_dims_valid, timestamp_dims_test = timestamp_dims[:valid_split_index], \
                                                                      timestamp_dims[
                                                                      valid_split_index:test_split_index], \
                                                                      timestamp_dims[test_split_index:]
    pred_exo_order_train, pred_exo_order_valid, pred_exo_order_test = get_ranking(pred_exo_idxs[:valid_split_index]), \
                                                        get_ranking(pred_exo_idxs[valid_split_index:test_split_index]), \
                                                        get_ranking(pred_exo_idxs[test_split_index:])
    train_len, valid_len, test_len = len(timestamps_train), len(timestamps_valid), len(timestamps_test)
    exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test = \
                                            [None]*train_len, [None]*valid_len, [None]*test_len
    timestamps_valid -= timestamps_valid[0]
    timestamps_test -= timestamps_test[0]
    train_num, valid_num, test_num = len(timestamps_train), len(timestamps_valid), len(timestamps_test)
    type_size = np.unique(mark).size
    dim = np.unique(timestamp_dims).size
    eventTrain = [list() for _ in range(dim)]
    eventTest = [list() for _ in range(dim)]
    timeTrain = [list() for _ in range(dim)]
    timeTest = [list() for _ in range(dim)]
    eventValid = [list() for _ in range(dim)]
    timeValid = [list() for _ in range(dim)]
    for i in range(len(mark_train)):
        u = timestamp_dims_train[i]
        exo_order_cascade_idx_train[pred_exo_order_train[i]] = (u, len(eventTrain[u]))
        _mark = mark_train[i]
        _time = timestamps_train[i]
        eventTrain[u] += [_mark]
        timeTrain[u] += [_time]
    for i in range(len(mark_valid)):
        u = timestamp_dims_valid[i]
        exo_order_cascade_idx_valid[pred_exo_order_valid[i]] = (u, len(eventValid[u]))
        _mark = mark_valid[i]
        _time = timestamps_valid[i]
        eventValid[u] += [_mark]
        timeValid[u] += [_time]
    for i in range(len(mark_test)):
        u = timestamp_dims_test[i]
        exo_order_cascade_idx_test[pred_exo_order_test[i]] = (u, len(eventTest[u]))
        _mark = mark_test[i]
        _time = timestamps_test[i]
        eventTest[u] += [_mark]
        timeTest[u] += [_time]
    train_seq_times = timeTrain
    train_seq_types = eventTrain
    valid_seq_times = timeValid
    valid_seq_types = eventValid
    test_seq_times = timeTest
    test_seq_types = eventTest
    for u in range(len(train_seq_times)):
        train_seq_times[u] = torch.tensor(train_seq_times[u])
        train_seq_types[u] = torch.tensor(train_seq_types[u])
        valid_seq_times[u] = torch.tensor(valid_seq_times[u])
        valid_seq_types[u] = torch.tensor(valid_seq_types[u])
        test_seq_times[u] = torch.tensor(test_seq_times[u])
        test_seq_types[u] = torch.tensor(test_seq_types[u])
    # process_list
    train_seq_times, train_seq_types, train_seq_lengths, train_tmax = process_list(train_seq_times, train_seq_types,
                                                                                   type_size)
    valid_seq_times, valid_seq_types, valid_seq_lengths, valid_tmax = process_list(valid_seq_times, valid_seq_types, type_size)
    test_seq_times, test_seq_types, test_seq_lengths, test_tmax = process_list(test_seq_times, test_seq_types,
                                                                               type_size)
    tmax = max([train_tmax, valid_tmax, test_tmax])

    return train_seq_times, valid_seq_times, test_seq_times, \
           train_seq_types, valid_seq_types, test_seq_types, \
           exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test, \
           train_seq_lengths, valid_seq_lengths, test_seq_lengths, \
           type_size, tmax, dim, train_num, valid_num, test_num


def pred(test_seq_times, test_seq_lengths, test_seq_types, model, type_size, tmax):
    incr_preds = np.empty((test_seq_times.shape[0], test_seq_times.shape[1]-1))
    mark_preds = np.empty((test_seq_times.shape[0], test_seq_times.shape[1]-1))
    incr_trues = np.array(test_seq_times[:, 1:] - test_seq_times[:, :-1])
    mark_trues = np.array(test_seq_types[:, 1:])
    for i in range(mark_trues.shape[1]):
        test_seq_lengths_u = torch.tensor([u if u <= (i+1) else (i+1) for u in test_seq_lengths])
        incr_estimates, incr_errors, types_real, types_estimates = prediction(device, model, test_seq_lengths_u,
                                                                              test_seq_times, test_seq_types,
                                                                              i, tmax)
        incr_preds[:, i] = incr_estimates
        mark_preds[:, i] = types_estimates

    valid_mask = mark_trues != type_size
    time_errors = np.abs(incr_preds - incr_trues)[valid_mask]
    mark_errors = (mark_preds != mark_trues)[valid_mask]
    return incr_preds, mark_preds, incr_trues, mark_trues, np.mean(time_errors), np.mean(mark_errors)


def get_endo_mask(cascade, cascade_exo_idx):
    endo_mask = np.full_like(cascade, True)
    for idx in cascade_exo_idx:
        endo_mask[idx[0]][idx[1]] = False
    return torch.tensor(endo_mask)


def train(train_exo_num, train_times_tensor, train_seq_types, train_seq_lengths,
                        valid_times_tensor, valid_seq_types, valid_seq_lengths,
                        test_times_tensor, test_seq_types, test_seq_lengths,
                        exo_order_cascade_idx_train, type_size, tmax):
    train_cascade_exo_idx = [exo_order_cascade_idx_train[i] for i in range(train_exo_num)]
    train_endo_mask = get_endo_mask(train_times_tensor, train_cascade_exo_idx)
    train_endo_mask = torch.tensor(train_endo_mask, dtype=torch.bool)
    with autograd.detect_anomaly():
        params = args, type_size, device, tmax, \
                 train_times_tensor, train_seq_types, train_seq_lengths, \
                 valid_times_tensor, valid_seq_types, valid_seq_lengths, \
                 test_times_tensor, test_seq_types, test_seq_lengths, train_endo_mask, \
                 args.batch_size, EPOCHS, USE_CUDA
        model = train_eval_sahp_exo(params)
    return model


def test(train_num, test_seq_times, time_valid_errors, mark_valid_errors, train_times_tensor, train_seq_types, train_seq_lengths,
                                                        valid_times_tensor, valid_seq_types, valid_seq_lengths,
                                                        test_times_tensor, test_seq_types, test_seq_lengths,
                                                        exo_order_cascade_idx_train, type_size, tmax):
    best_time_exo_ratio, best_mark_exo_ratio = exo_ratios[np.argmin(time_valid_errors)], exo_ratios[np.argmin(mark_valid_errors)]
    best_time_train_exo_num, best_mark_train_exo_num = int(train_num * best_time_exo_ratio), \
                                                       int(train_num * best_mark_exo_ratio)
    model1 = train(best_time_train_exo_num, train_times_tensor, train_seq_types, train_seq_lengths,
                  valid_times_tensor, valid_seq_types, valid_seq_lengths,
                  test_times_tensor, test_seq_types, test_seq_lengths,
                  exo_order_cascade_idx_train, type_size, tmax)
    model2 = train(best_mark_train_exo_num, train_times_tensor, train_seq_types, train_seq_lengths,
                        valid_times_tensor, valid_seq_types, valid_seq_lengths,
                        test_times_tensor, test_seq_types, test_seq_lengths,
                        exo_order_cascade_idx_train, type_size, tmax)
    _, _, _, _, time_error1, mark_error1 = \
        pred(test_seq_times, test_seq_lengths, test_seq_types, model1, type_size, tmax)
    _, _, _, _, time_error2, mark_error2 = \
        pred(test_seq_times, test_seq_lengths, test_seq_types, model2, type_size, tmax)
    return np.min((time_error1, time_error2)), np.min((mark_error1, mark_error2))


def sahp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark):
    train_seq_times, valid_seq_times, test_seq_times, \
    train_seq_types, valid_seq_types, test_seq_types, \
    exo_order_cascade_idx_train, exo_order_cascade_idx_valid, exo_order_cascade_idx_test, \
    train_seq_lengths, valid_seq_lengths, test_seq_lengths, \
    type_size, tmax, dim, train_num, valid_num, test_num = \
        sahp_data_preprocess(timestamps, timestamp_dims, mark, pred_exo_idxs)
    # Define training data
    train_times_tensor, train_seq_types, train_seq_lengths = train_seq_times.to(device), \
                                                             train_seq_types.to(device), train_seq_lengths.to(device)
    # Define development data
    valid_times_tensor, valid_seq_types, valid_seq_lengths = valid_seq_times.to(device), \
                                                             valid_seq_types.to(device), valid_seq_lengths.to(device)
    # Define test data
    test_times_tensor, test_seq_types, test_seq_lengths = test_seq_times.to(device), \
                                                          test_seq_types.to(device), test_seq_lengths.to(device)
    # find a best ratio through valid set
    time_valid_errors, mark_valid_errors = [], []
    for exo_ratio in exo_ratios:
        train_exo_num = int(train_num * exo_ratio)
        model = train(train_exo_num, train_times_tensor, train_seq_types, train_seq_lengths,
                        valid_times_tensor, valid_seq_types, valid_seq_lengths,
                        test_times_tensor, test_seq_types, test_seq_lengths,
                        exo_order_cascade_idx_train, type_size, tmax)
        _, _, _, _, time_error, mark_error = \
            pred(valid_seq_times, valid_seq_lengths, valid_seq_types, model, type_size, tmax)
        time_valid_errors += [time_error]
        mark_valid_errors += [mark_error]

    return test(train_num, test_seq_times, time_valid_errors, mark_valid_errors,
                                                        train_times_tensor, train_seq_types, train_seq_lengths,
                                                        valid_times_tensor, valid_seq_types, valid_seq_lengths,
                                                        test_times_tensor, test_seq_types, test_seq_lengths,
                                                        exo_order_cascade_idx_train, type_size, tmax)
