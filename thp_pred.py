from tensorflow.contrib.keras import preprocessing
import torch.nn as nn
import torch.optim as optim
from model.non_linear.thp.Main import train_with_exo_mae, eval_epoch_mae
from model.non_linear.thp.transformer.Models import Transformer
from model.non_linear.thp import Utils
from model.non_linear.thp.preprocess.Dataset import get_masked_dataloader
from func.common import *

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


def train_with_exo(trainloader, pred_loss_func, num_types):
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
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    train_with_exo_mae(model, trainloader, optimizer, scheduler, pred_loss_func, opt)
    return model


def thp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark, k=1.):
    # data and params
    train_rate, test_rate = train_test_ratio
    test_split_index = int(train_rate * len(timestamps))
    timestamps_train, timestamps_test = timestamps[:test_split_index], timestamps[test_split_index:]
    mark_train, mark_test = mark[:test_split_index], mark[test_split_index:]
    timestamp_dims_train, timestamp_dims_test = timestamp_dims[:test_split_index], timestamp_dims[test_split_index:]
    timestamps_test -= timestamps_test[0]

    exo_num = int(len(timestamps_train) * k)
    train_exo_idxs = pred_exo_idxs[pred_exo_idxs < test_split_index]
    exo_idxs = train_exo_idxs[:exo_num]
    endo_mask_train = np.full_like(timestamps_train, True)
    endo_mask_train[exo_idxs] = False
    dim = np.unique(timestamp_dims).size
    num_types = np.unique(mark).size
    train_data, _ = process_masked_data(timestamps_train, timestamp_dims_train, mark_train, endo_mask_train, dim)
    test_data, _ = process_masked_data(timestamps_test, timestamp_dims_test, mark_test, endo_mask_train, dim)
    # trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    # testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    trainloader = get_masked_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_masked_dataloader(test_data, opt.batch_size, shuffle=False)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    num_types = np.unique(mark).size
    model = train_with_exo(trainloader, pred_loss_func, num_types)

    # evaluation
    pred_event, true_event, pred_time, true_time, likelihood, mark_error, time_error = \
        eval_epoch_mae(model, testloader, pred_loss_func, opt)

    return pred_event, true_event, pred_time, true_time, likelihood, mark_error, time_error
