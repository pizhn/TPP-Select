from tensorflow.contrib.keras import preprocessing
import torch.nn as nn
import torch.optim as optim
from model.non_linear.thp.Main import train_mae
from model.non_linear.thp.transformer.Models import Transformer
from model.non_linear.thp import Utils
from model.non_linear.thp.preprocess.Dataset import get_dataloader
from func.constants import *
from predict.common import *

pad_sequences = preprocessing.sequence.pad_sequences


def process_masked_data(timestamps, timestamp_dims, mark, endo_mask, dim):
    data = [list() for _ in range(dim)]
    for t, u, msk, m in zip(timestamps, timestamp_dims, endo_mask, mark):
        if len(data[u]) == 0:
            time_since_last_event = 0
        else:
            time_since_last_event = t - data[u][-1]['time_since_start']
        data[u] += [{'time_since_start': t,
                     'time_since_last_event': time_since_last_event,
                     'type_event': m,
                     'endo_mask': msk}]
    data = [d for d in data if len(d) > 0]
    return data, np.unique(mark).size


def test(exo_num, pred_exo_idxs, timestamps_train, timestamp_dims_train,
         mark_train, timestamps_test, timestamp_dims_test, mark_test, endo_mask_train, dim, num_types, label):
    model = train_eval(exo_num, pred_exo_idxs, timestamps_train, \
                                        timestamp_dims_train, mark_train, \
                                        timestamps_test, timestamp_dims_test, mark_test, endo_mask_train, dim, \
                                        num_types, label)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    pred_event, true_event, _, _, _, mark_error, time_error = train_mae(model, trainloader, validloader, optimizer,
                                                                        scheduler, pred_loss_func, opt)
    return time_error, mark_error


def train_eval(k, pred_exo_idxs, timestamps_train, timestamp_dims_train, mark_train, timestamps_valid,
               timestamp_dims_valid, mark_valid, dim, num_types):
    exo_num = int(len(timestamps_train * k))
    exo_idxs = pred_exo_idxs[:exo_num]
    endo_mask_train = np.full_like(timestamps_train, True)
    endo_mask_train[exo_idxs] = False
    train_data, _ = process_masked_data(timestamps_train, timestamp_dims_train, mark_train, endo_mask_train, dim)
    valid_data, _ = process_masked_data(timestamps_valid, timestamp_dims_valid, mark_valid, endo_mask_train, dim)
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    validloader = get_dataloader(valid_data, opt.batch_size, shuffle=False)
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)
    return model


def thp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark, label, k=1.):
    # process data
    train_rate, test_rate = train_test
    test_split_index = int(train_rate * len(timestamps))
    timestamps_train, timestamps_test = timestamps[:test_split_index], timestamps[test_split_index:]
    mark_train, mark_test = mark[:test_split_index], mark[test_split_index:]
    timestamp_dims_train, timestamp_dims_test = timestamp_dims[:test_split_index], timestamp_dims[test_split_index:]
    timestamps_test -= timestamps_test[0]
    dim = np.unique(timestamp_dims).size
    num_types = np.unique(mark).size
    return test(k, pred_exo_idxs, timestamps_train, timestamp_dims_train,
                mark_train, timestamps_test, timestamp_dims_test, mark_test, dim, num_types, label)
