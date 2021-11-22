from NODE.models import *
import numpy as np
from torch.nn import functional as F
from Utilities.Solvers import *
import tqdm


def compute_validation_loss(ode_train, val_path, step_skip):
    val_set = np.load(val_path)
    val_len, *r = val_set.shape
    val_set = val_set.reshape([val_len, -1])
    val_set = Tensor(val_set[:, :]).detach().unsqueeze(1)

    _, _, state_dim = val_set.shape
    val_times = Tensor(np.arange(val_len)).detach() * step_skip  # 0 2 4 6 8
    with torch.no_grad():
        traj_segments_list = []
        all_init = []
        for j in range(0, val_len - 1, 1):
            val_time_segment = val_times[j:j + 2]
            val_traj_segment = val_set[j:j + 2]
            init_con = val_traj_segment[0]
            all_init.append(init_con)
            traj_segments_list.append(val_traj_segment)
        # concatenating all batches together
        all_init = torch.cat(all_init).view(-1, 1, np.prod(r))
        z1 = ode_train(all_init.to(device), val_time_segment.to(device), return_whole_sequence=True).squeeze()
        obs1 = torch.cat(traj_segments_list, 1).to(device)
        loss = F.mse_loss(z1[:, :, -2:], obs1[:, :, -2:])
        return loss.item()


def sample_and_grow(ode_train, traj_list, epochs, LR, lookahead, plot_freq=50,
                    save_path=None, step_skip=1):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ode_train.parameters()), lr=LR)
    for i in tqdm.tqdm(range(epochs), desc="Training progress", position=0, leave=True):
        for idx, true_traj in enumerate(traj_list):
            n_segments, _, n_state = true_traj.size()
            true_segments_list = []

            for j in range(0, n_segments - lookahead + 2 - BATCH_SKIP, BATCH_SKIP):
                j = j + i % BATCH_SKIP
                true_sampled_time_segment = (Tensor(np.arange(lookahead)) * step_skip).detach()
                true_sampled_segment = true_traj[j:j + lookahead]
                true_segments_list.append(true_sampled_segment)
            # concatenating all batches together
            obs = torch.cat(true_segments_list, 1).to(device)

            pred_traj = []
            ode_input = obs[0, :, :].unsqueeze(1).to(device)  # initial condition has size [1499, 1, 17]
            pred_traj.append(ode_input)
            for k in range(len(true_sampled_segment) - 1):
                # ode input shape:  torch.Size([time_len, 1, state_dim])
                # print("input shape: ", ode_input.shape)
                z1 = ode_train(ode_input, Tensor(np.arange(2)).to(device)).squeeze(1)
                ode_input = torch.cat([obs[k + 1, :, :-2].unsqueeze(1),  # concatenating uv_vel and latlon
                                       z1[:, -2:].unsqueeze(1)], 2)
                pred_traj.append(ode_input)

            # prediction has size [plot_len, 1, 17]
            pred_traj = torch.swapaxes(torch.cat(pred_traj, 1), 0, 1)

            if idx == 0:
                # loss from the first trajectory, only on latlon
                loss = F.mse_loss(pred_traj[:, :, -2:], obs[:, :, -2:])
            else:
                # adding loss from other trajectories
                loss += F.mse_loss(pred_traj[:, :, -2:], obs[:, :, -2:])

        train_loss_arr.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # validation
        val_loss = compute_validation_loss(ode_train, "Data/training_data/with_3days_flow/train_data_nowcast_drifter_2_knn_10.npy",
                                           step_skip)
        val_loss_arr.append(val_loss)

        if i % plot_freq == 0:
            if save_path is not None:
                torch.save({'ode_train': ode_train, 'train_loss_arr': train_loss_arr, 'val_loss_arr': val_loss_arr},
                           save_path)

            # computing trajectory using the current model
            plot_title = "\nIteration: {0} Step Size: {1} No. of Points: {2} Lookahead: {3} LR: {4}\n    Training Loss: {5:.3e}\n    Validation Loss: {6:.3e}"
            print(plot_title.format(i, step_size, n_segments, LOOKAHEAD, LR, loss.item(), val_loss))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_data_path_list = [#"Data/training_data/train_data_nowcast_drifter_41_knn_10.npy",
                               #"Data/training_data/train_data_nowcast_drifter_52_knn_10.npy",
                               #"Data/training_data/train_data_nowcast_drifter_59_knn_10.npy",
                               "Data/training_data/with_3days_flow/train_data_nowcast_drifter_59_knn_10.npy"]
                               #"Data/training_data/train_data_nowcast_drifter_59_knn_10.npy"]
    train_traj_list = []
    # concatenating all trajectories into a list
    for i, data_path in enumerate(training_data_path_list):
        with open(data_path, 'rb') as f:
            train_set = np.load(f)
            print("Training traj {0} has shape: \n".format(i), train_set.shape)
        full_len = train_set.shape[0]
        train_len = 336
        train_set = Tensor(train_set.reshape([full_len, -1]))[:train_len]
        train_traj = train_set.detach().unsqueeze(1)
        # train_traj = train_traj + torch.randn_like(train_traj) * 0.00
        train_traj_list.append(train_traj)

    # KNODE parameters
    ode_solve = Euler
    step_size = 0.002
    torch.manual_seed(0)
    # ode_train = torch.load("SavedModels/full_train_on_2traj.pth")['ode_train']
    ode_train = NeuralODE(WaveRiders3Day(device), ode_solve, step_size).to(device)

    # training parameters
    step_skip = 1  # number of interpolations between observations
    train_loss_arr = []
    val_loss_arr = []
    save_path = 'NODE/saved_models/drifter_59_tanh_3day.pth'
    BATCH_SKIP = 1
    EPOCHs = 4000  # No. of epochs to optimize
    LOOKAHEAD = 2  # alpha, the number of steps to lookahead
    name = "lookahead_" + str(LOOKAHEAD - 1)
    LR = 0.001  # optimization step size
    plot_freq = 20

    sample_and_grow(ode_train, train_traj_list, EPOCHs, LR, LOOKAHEAD,
                    plot_freq=plot_freq, save_path=save_path, step_skip=step_skip)