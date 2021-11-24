import os
import matplotlib.pyplot as plt
from matplotlib import animation

def vis_vf(i, xxyy, gt_vf, pred_vf, name='SantaBarbara', anim_folder='./VFAnim'):
    '''

    :param gt_vf: [t_vis, n_x_grid, n_y_grid, 4(U,V,x,y)]
    :param pred_vf: [t_vis, n_x_grid, n_y_grid, 4(U,V,x,y)]
    :return:
    '''
    # convert the tensor to numpy array
    gt_vf = gt_vf.to('cpu').numpy()
    pred_vf = pred_vf.to('cpu').detach().numpy()
    xxyy = xxyy.to('cpu').numpy()
    # fix the xx,yy grid mesh
    xx = xxyy[:, :, 0]
    yy = xxyy[:, :, 1]
    # gt U&V, pred U&V
    gt_U = gt_vf[:, :, :, 0]
    gt_V = gt_vf[:, :, :, 1]
    pred_U = pred_vf[:, :, :, 0]
    pred_V = pred_vf[:, :, :, 1]

    ## Animate
    fig = plt.figure()
    fig.set_size_inches(20, 10)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    # fig, ax = plt.subplots()
    # ax1 = plt.subplot()
    ## init the animation frame
    gt_U0, gt_V0 = gt_U[0, :, :], gt_V[0, :, :]
    pred_U0, pred_V0 = pred_U[0, :, :], pred_V[0, :, :]
    gt_Q = ax1.quiver(xx, yy, gt_U0, gt_V0, color='black', label='GT Flow')
    pred_Q = ax2.quiver(xx, yy, pred_U0, pred_V0, color='red', label='Pred Flow')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title('Epoch:'+str(i)+' '+name)

    def update_quiver(ti):
        # print(ti)
        gt_Ui, gt_Vi = gt_U[ti, :, :], gt_V[ti, :, :]
        pred_Ui, pred_Vi = pred_U[ti, :, :], pred_V[ti, :, :]
        gt_Q.set_UVC(gt_Ui, gt_Vi)
        pred_Q.set_UVC(pred_Ui, pred_Vi)
        return gt_Q, pred_Q

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, frames=gt_U.shape[0])
    save_name = os.path.join(anim_folder, name+'_epoch'+str(i)+'.mp4')
    anim.save(save_name, writer=animation.FFMpegWriter(fps=10))


def vis_vf_large(i, gt_vf, forecast_vf, pred_vf, name='SantaBarbara', anim_folder='./VFAnim'):
    '''

    :param gt_vf: [t_vis, n_x_grid, n_y_grid, 4(U,V)]
    :param pred_vf: [t_vis, n_x_grid, n_y_grid, 4(U,V)]
    :return:
    '''
    # convert the tensor to numpy array
    gt_vf = gt_vf.to('cpu').numpy()
    forecast_vf = forecast_vf.to('cpu').numpy()
    pred_vf = pred_vf.to('cpu').detach().numpy()

    # gt U&V, pred U&V
    gt_U = gt_vf[:, :, :, 0]
    gt_V = gt_vf[:, :, :, 1]
    forecast_U = forecast_vf[:, :, :, 0]
    forecast_V = forecast_vf[:, :, :, 1]
    pred_U = pred_vf[:, :, :, 0]
    pred_V = pred_vf[:, :, :, 1]

    ## Animate
    fig = plt.figure()
    fig.set_size_inches(30, 10)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    # fig, ax = plt.subplots()
    # ax1 = plt.subplot()
    ## init the animation frame
    gt_U0, gt_V0 = gt_U[0, :, :], gt_V[0, :, :]
    forecast_U0, forecast_V0 = forecast_U[0, :, :], forecast_V[0, :, :]
    pred_U0, pred_V0 = pred_U[0, :, :], pred_V[0, :, :]
    gt_Q = ax1.imshow(gt_U0, cmap='ocean', interpolation='nearest')
    forecast_Q = ax2.imshow(forecast_U0, cmap='ocean', interpolation='nearest')
    pred_Q = ax3.imshow(pred_U0, cmap='ocean', interpolation='nearest')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title('Epoch:'+str(i)+' '+name)
    plt.close()

    def update_quiver(ti):
        # print(ti)
        gt_Ui, gt_Vi = gt_U[ti, :, :], gt_V[ti, :, :]
        forecast_Ui, forecast_Vi = forecast_U[ti, :, :], forecast_V[ti, :, :]
        pred_Ui, pred_Vi = pred_U[ti, :, :], pred_V[ti, :, :]
        gt_Q.set_data(gt_Ui)
        forecast_Q.set_data(forecast_Ui)
        pred_Q.set_data(pred_Ui)
        return gt_Q, pred_Q

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, frames=gt_U.shape[0])
    save_name = os.path.join(anim_folder, name+'_epoch'+str(i)+'.mp4')
    anim.save(save_name, writer=animation.FFMpegWriter(fps=10))