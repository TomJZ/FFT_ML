import numpy as np
import matplotlib.pyplot as plt
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


today = date.today().strftime("%m%d%Y")


def trainer(model, train_loader, val_loader, optimizer,
            EPOCHs, LR, LOOKAHEAD, BATCH_SKIP, time_aug,
            loss_arr, val_loss_arr,
            plot_freq=50, save_path=None, device=None):

    # running loss for log purpose
    run_train_loss = 0

    for i in tqdm.tqdm(range(EPOCHs)):
        if i > 200:
            optimizer.param_groups[0]['lr'] = LR/10
        for idx, data in enumerate(train_loader):
            si, ti = [data_i.to(device) for data_i in data]

            # pre-process the trajectory into segments
            xxyy = si[0,0,:,:,2:]
            si = si[:,:,:,:,:2]
            ti = (ti - ti[:, 0]).detach() * time_aug  # remove the offset in time
            segs_i, init_conds_t, ts_i = seg_traj(i, si, ti, LOOKAHEAD, BATCH_SKIP)

            # model prediction
            zi = model(init_conds_t, ts_i, return_whole_sequence=True).squeeze()

            # compute MSE loss
            optimizer.zero_grad()
            loss = F.mse_loss(zi, segs_i)
            loss_arr.append(loss.item())
            run_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # log & visualization
        with torch.no_grad():
            if i % plot_freq == 0:
                ## log the loss
                avg_run_train_loss = run_train_loss if i==0 else run_train_loss / plot_freq  # avg the loss
                run_train_loss = 0  # refresh the loss
                # print log
                print(avg_run_train_loss)

                ## visualize
                vis_batch_index = 0
                vis_len = 50
                pred_lag = 10
                segs_vis, init_conds_vis, ts_vis = seg_traj(i, si[vis_batch_index].unsqueeze(0), ti[vis_batch_index].unsqueeze(0),
                                                            pred_lag, pred_lag)
                # predict the whole sequence
                # t_vis = ti[vis_batch_index, :vis_len]  # [vis_len]
                # si_vis = si[vis_batch_index, :vis_len]
                zi_vis = model(init_conds_vis, ts_vis, return_whole_sequence=True).squeeze(1)  # pred vis_len sequence given model [pred_lag, n_seg, x_grid, y_grid, dim(2)]
                # visualize the vector field
                si_vis_flat = segs_vis.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]
                zi_vis_flat = zi_vis.permute(1, 0, 2, 3, 4).flatten(end_dim=1)[:vis_len]

                # save the animation
                # vis_vf(i, xxyy, si_vis_flat, zi_vis_flat)

                # compute the draft between the ground truth and prediction
                draft_error(i, zi_vis, segs_vis)  # compute the draft error given the current trained model



if __name__ == '__main__':
    ## fix the random seed
    seed = 0
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

    ## get the training data
    train_data_folder = './data/train'
    val_data_folder = './data/val'

    ## set up the dataloader
    # build torch dataset given the dataset folder
    train_dataset = Dataset_builder(train_data_folder)
    val_dataset = Dataset_builder(val_data_folder)
    # build torch dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    ## training & eval
    # training settings
    loss_arr = []
    val_loss_arr = []
    EPOCHs = 1000  # No. of epochs to optimize
    BATCH_SKIP = 1  # LOOKAHEAD-1
    LOOKAHEAD = 4  # number of observations to lookahead for data augmentation
    time_aug = 4

    # NODE parameters
    ode_solve = Euler
    step_size = 0.01
    LR = 0.01  # optimization step size
    # model init
    node_flow_model = NeuralODE(FlowNN(device), ode_solve, step_size).to(device)
    # optimizer init
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, node_flow_model.parameters()), lr=LR)

    # training
    save_path = './experiments/saved_models/' + today + '/flow_model.pth'
    trainer(node_flow_model, train_loader, val_loader, optimizer,
            EPOCHs, LR, LOOKAHEAD, BATCH_SKIP, time_aug,
            loss_arr, val_loss_arr,
            plot_freq=50, save_path=save_path, device=device)
