import torch
import matplotlib.pyplot as plt
def draft_error(i, gt_flow, forecast_flow, pred_flow, display_lag=(10,20),save_name="region_error_plot"):
    '''
    :param pred_flow: [pred_lag, n_seg, x_grid, y_grid, dim(2)]
    :param gt_flow: [pred_lag, n_seg, x_grid, y_grid, dim(2)]
    :return: draft_error_arr: [pred_lag, n_seg]
    '''
    fig = plt.figure()
    dist_error_gt_pred = torch.sqrt(torch.sum((pred_flow - gt_flow), dim=-1)**2)  # [pred_lag, n_seg, x_grid, y_grid] distance error at each grid
    draft_error_arr_gt_pred = torch.sum(dist_error_gt_pred, dim=[1, 2, 3])  # [pred_lag]
    plt.plot(draft_error_arr_gt_pred.cpu()[display_lag[0]:display_lag[1]], label="pred draft")

    dist_error_gt_forecast = torch.sqrt(
        torch.sum((forecast_flow - gt_flow), dim=-1) ** 2)  # [pred_lag, n_seg, x_grid, y_grid] distance error at each grid
    draft_error_arr_gt_forecast = torch.sum(dist_error_gt_forecast, dim=[1, 2, 3])  # [pred_lag]
    plt.plot(draft_error_arr_gt_forecast.cpu()[display_lag[0]:display_lag[1]], label="forecast draft")


    plt.xlabel('lag time')
    plt.ylabel('Distance of draft add up on each segments at all grids')
    plt.title('Epoch:'+str(i)+' - Draft Error Plot')
    # plt.ylim([0, 8000])
    plt.legend()
    plt.savefig('./Metrics/'+save_name+'_epoch'+str(i)+'.png')
    # plt.show()

    pass