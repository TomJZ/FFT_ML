import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from Utilities.Solvers import *
from NODE.NODE import *
from systems_flow import *
import sys
import random
from datetime import date
from Utilities.DataLoader import *
from torch.utils.data import DataLoader
from matplotlib import animation
import os
from Utilities.Metrics import *
from Utilities.Preprocess import *
from Utilities.Animation import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.cuda.amp as amp


today = '11202021'

def norm(data):
    data[data > 10e10] = 0
    means = data.mean(dim=1, keepdim=True)
    stds = data.std(dim=1, keepdim=True)
    norm_data = (data - means) / stds
    norm_data = torch.nan_to_num(norm_data)
    return norm_data, means, stds

def denorm(data, means, stds):
    denorm_data = data*stds + means
    return denorm_data


def infer(model, train_loader, eval_loader, forecast_loader, save_name, epoch):
    ## load the ground truth and forecast dataset
    for idx, data in enumerate(train_loader):
        si_train, ti_train = [data_i.to(device) for data_i in data]
        si_train[si_train > 10e10] = 0
        si_train_lat_lon = si_train[:, :, :, :, :2]
        si_train_uv = si_train[:, :, :, :, 2:]

    for idx, data in enumerate(eval_loader):
        si_eval, ti_eval = [data_i.to(device) for data_i in data]
        si_eval[si_eval > 10e10] = 0
        si_eval_lat_lon = si_eval[:, :, :, :, :2]
        si_eval_uv = si_eval[:, :, :, :, 2:]

    for idx, data in enumerate(forecast_loader):
        si_forecast, ti_forecast = [data_i.to(device) for data_i in data]
        si_forecast[si_forecast > 10e10] = 0
        si_forecast_lat_lon = si_forecast[:, :, :, :, :2]
        si_forecast_uv = si_forecast[:, :, :, :, 2:]

    with torch.no_grad():
        model.eval()

        ## visualize
        vis_len = 72

        # predict the whole sequence
        # context points
        si_pred_ctx = si_train_uv.to(device)
        si_gt = si_eval_uv

        # Transformer inference (loop)
        init_target = si_pred_ctx[:, -1:]
        zik = init_target  # changing target point
        zik_vis = zik  # extended vis tensor
        for pred_index in range(vis_len):
            # make prediction mask
            si_src_masks, si_tgt_masks, si_memory_masks = create_mask(src_len=si_pred_ctx.shape[1], tgt_len=1,
                                                                      DEVICE=device)
            # prediction forward
            zik_next = model(si_pred_ctx, zik, si_src_masks, si_tgt_masks, si_memory_masks)[:, -1:]
            si_pred_ctx = torch.cat([si_pred_ctx, zik_next], dim=1)
            zik = zik_next
            zik_vis = torch.cat([zik_vis, zik_next], dim=1)
        zi_vis = zik_vis[:, 1:]

        # visualize the vector field
        si_gt_flat = si_gt.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]
        si_forecast_flat = si_forecast_uv.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]
        zi_vis_flat = zi_vis.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]

        # save the animation
        vis_vf_large(epoch, si_gt_flat, si_forecast_flat, zi_vis_flat, save_name)

        # compute the draft error between the ground truth and prediction
        draft_error(epoch, si_gt.permute(1, 0, 2, 3, 4), si_forecast_uv.permute(1, 0, 2, 3, 4), zi_vis.permute(1, 0, 2, 3, 4),
                    display_lag=(0, 73), save_name=save_name)  # compute the draft error given the current trained model

        # torch.save(torch.cat([si_eval_lat_lon, zi_vis], dim=-1).squeeze(0), './PredictionTensor/predict_'+save_name+'.pth')
        # torch.save(si_eval.squeeze(0), './PredictionTensor/eval_'+save_name+'.pth')
        # torch.save(si_forecast.squeeze(0), './PredictionTensor/forecast_'+save_name+'.pth')


pass

def trainer(model, train_loader, eval_loader, forecast_loader, optimizer,
            EPOCHs, LR, LOOKAHEAD, BATCH_SKIP, time_aug, train_bs, context_pts, forecast_pts,
            loss_arr, val_loss_arr,
            plot_freq=50, save_path=None, save_name=None, device=None):

    # running loss for log purpose
    run_train_loss = 0

    # setting up the training dataset
    for idx, data in enumerate(train_loader):
        si, ti = [data_i for data_i in data]
        si_train = si[:, :, :, :, 2:]
        si_train[si_train > 10e10] = 0  # filter out the invalid data

        # cut trajectory into segments
        segs_i, init_conds_t, ts_i = seg_traj(si_train, ti, LOOKAHEAD, BATCH_SKIP)


    for i in tqdm.tqdm(range(EPOCHs)):
        if i > 7000:
            optimizer.param_groups[0]['lr'] = LR/10
        for idx, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            # random sample index
            batch_index = np.random.choice(si_train.shape[1]-LOOKAHEAD+1, train_bs)
            # construct the training dataset, source data, target input data, target output data
            si_src_ts = segs_i.permute(1, 0, 2, 3, 4)[batch_index]
            si_src_context_ts = si_src_ts[:, :context_pts].to(device)
            si_tgt_input_ts = si_src_ts[:, context_pts - 1:-1].to(device)
            si_tgt_output_ts = si_src_ts[:, context_pts:].to(device)

            # create the mask for
            si_src_masks, si_tgt_masks, si_memory_masks = create_mask(src=si_src_context_ts, tgt=si_tgt_input_ts, DEVICE=device)
            # model prediction
            zi = model(si_src_context_ts, si_tgt_input_ts, si_src_masks, si_tgt_masks, si_memory_masks)

            # compute loss
            # optimizer.zero_grad()
            loss_func = nn.HuberLoss()
            loss = loss_func(zi, si_tgt_output_ts)
            loss_arr.append(loss.item())
            run_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # log & visualization
        with torch.no_grad():
            ## load the ground truth and forecast dataset
            for idx, data in enumerate(eval_loader):
                si_eval, ti_eval = [data_i.to(device) for data_i in data]
                si_eval = si_eval[:,:,:,:,2:]
                si_eval[si_eval > 10e10] = 0

            for idx, data in enumerate(forecast_loader):
                si_forecast, ti_forecast = [data_i.to(device) for data_i in data]
                si_forecast = si_forecast[:,:,:,:,2:]
                si_forecast[si_forecast > 10e10] = 0

            if i % plot_freq == 0:
                model.eval()

                ## save model
                torch.save({'model': model, 'loss_arr': loss_arr}, save_path+"_epoch"+str(i)+".pth")

                ## log the loss
                avg_run_train_loss = run_train_loss if i==0 else run_train_loss / plot_freq  # avg the loss
                run_train_loss = 0  # refresh the loss
                # print log
                print(avg_run_train_loss)

                ## visualize
                vis_len = 72

                # predict the whole sequence
                # context points
                si_pred_ctx = si_train.to(device)
                si_gt = si_eval


                # Transformer inference (loop)
                init_target = si_pred_ctx[:, -1:]
                zik = init_target  # changing target point
                zik_vis = zik  # extended vis tensor
                for pred_index in range(vis_len):
                    # make prediction mask
                    si_src_masks, si_tgt_masks, si_memory_masks = create_mask(src_len=si_pred_ctx.shape[1], tgt_len=1,
                                                                              DEVICE=device)
                    # prediction forward
                    zik_next = model(si_pred_ctx, zik, si_src_masks, si_tgt_masks, si_memory_masks)[:, -1:]
                    si_pred_ctx = torch.cat([si_pred_ctx, zik_next], dim=1)
                    zik = zik_next
                    zik_vis = torch.cat([zik_vis, zik_next], dim=1)
                zi_vis = zik_vis[:, 1:]

                # visualize the vector field
                si_gt_flat = si_gt.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]
                si_forecast_flat = si_forecast.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]
                zi_vis_flat = zi_vis.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]

                # save the animation
                xxyy = si[0, 0, :, :, :2]
                vis_vf_large(i, si_gt_flat, si_forecast_flat, zi_vis_flat, save_name)

                # compute the draft error between the ground truth and prediction
                draft_error(i, si_gt.permute(1,0,2,3,4), si_forecast.permute(1,0,2,3,4), zi_vis.permute(1,0,2,3,4), display_lag=(0, 73), save_name=save_name)  # compute the draft error given the current trained model



if __name__ == '__main__':
    ## fix the random seed
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # specify the device being used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # auto-selection
    # device = "cpu"  # or just manually specify here


    # Load region of Interests
    _, lat_min, lon_min = sys.argv
    lat_min = int(lat_min)
    lat_max = lat_min + 50
    lon_min = int(lon_min)
    lon_max = lon_min + 100

    # lat_min = 200
    # lat_max = 250
    # lon_min = 0
    # lon_max = 100

    print('lat_min:', lat_min)
    print('lat_max:', lat_max)
    print('lon_min:', lon_min)
    print('lon_max:', lon_max)

    ## get the training data
    train_data_folder = './data/train'
    eval_data_folder = './data/eval'
    forecast_data_folder = './data/forecast'

    ## set up the dataloader
    # build torch dataset given the dataset folder
    train_dataset = Dataset_builder(train_data_folder, "train", lat_min, lat_max, lon_min, lon_max)
    eval_dataset = Dataset_builder(eval_data_folder, "eval", lat_min, lat_max, lon_min, lon_max)
    forecast_dataset = Dataset_builder(forecast_data_folder, "forecast", lat_min, lat_max, lon_min, lon_max)

    # build torch dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    forecast_loader = DataLoader(forecast_dataset, batch_size=1, shuffle=False, num_workers=0)

    ## training & eval
    # training settings
    loss_arr = []
    val_loss_arr = []
    EPOCHs = 10001  # No. of epochs to optimize
    BATCH_SKIP = 1  # LOOKAHEAD-1
    LOOKAHEAD = 300  # number of observations to lookahead for data augmentation
    context_pts = 150
    forecast_pts = LOOKAHEAD - context_pts
    time_aug = 4

    # Trainer parameters
    train_bs = 10
    LR = 0.0001  # optimization step size

    # model setting
    s_dim = 2106
    d_model = 512
    nhead = 16
    d_hid = 2048
    nlayers = 2
    dropout = 0.3

    # training
    save_name = 'lat_' + str(lat_min) + '_' + str(lat_max) \
                + '_lon_' + str(lon_min) + '_' + str(lon_max)
    save_path = './experiments/' + today + '/saved_models/flow_model_' \
                + save_name
    mode = "Train"

    if mode == "Train":
        # model init
        node_flow_model = TransformerModel(s_dim, d_model, nhead, d_hid, nlayers, dropout).to(device)
    elif mode == "Eval":
        epoch = 10000
        # load model
        node_flow_model = torch.load(save_path+'_epoch'+str(epoch)+'.pth')["model"]

    # optimizer init
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, node_flow_model.parameters()), lr=LR)

    if mode == "Train":
        trainer(node_flow_model, train_loader, eval_loader, forecast_loader, optimizer,
                EPOCHs, LR, LOOKAHEAD, BATCH_SKIP, time_aug, train_bs, context_pts, forecast_pts,
                loss_arr, val_loss_arr,
                plot_freq=1000, save_path=save_path, save_name=save_name, device=device)

    # see if the model could overfit
    elif mode == "Eval":
        infer(node_flow_model, train_loader, eval_loader, forecast_loader, save_name=save_name, epoch=epoch)