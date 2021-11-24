import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib import animation, rc, rcParams
mpl.rcParams['legend.fontsize'] = 22
def to_np(x):
    return x.detach().cpu().numpy()


'''
TODO: proper scaling for communication range
@brief: Making animation for true and predicted trajectories of 3D robot swarms

@param true_traj: The true trajectory of a swarm, shape must be [num_steps x num_aganets x 3]
@param true_vel: The true velocity (or orientation) of a swarm, shape must be [num_steps x num_aganets x 3]
@param pred_traj: The predicted trajectory of a swarm, shape must be [num_steps x num_aganets x 3]
@param pred_vel: The predicted velocity (or orientation) of a swarm, shape must be [num_steps x num_aganets x 3]
@param total_frame: The total number of frames to animate, if total_frame>num_step, then total_frame = num_steps
@param save_name: The name of mp4 to be saved
@param include_true: whether true trajectories are included
@param include_pred: whether predicted trajectories are included
@param Compare: whether to draw a line between true agents and predicted agents
@param Trailing: whether to draw the trailing edge, this applies to both true and pred
@param Comm: whether to include communication visualization, only applies to prediction
@param save: whether to save the animation
'''

def make_animation(true_traj, true_vel, pred_traj, pred_vel, total_frame, save_name, percept_range, box_size=5,
                   include_true=True, include_pred=False, Compare=False, Trailing=False, Comm=False, save=0):
  def scale_marker_size(range):
    return range*65
  
  import mpl_toolkits.mplot3d.axes3d as p3
  mpl.rcParams['animation.embed_limit'] = 2**128
  # Making legends
  blue_dot = mlines.Line2D([], [], color='dodgerblue', marker='.', linestyle='None',
                            markersize=15, label='Prediction')
  red_dot = mlines.Line2D([], [], color='orangered', marker='.', linestyle='None',
                            markersize=15, label='Ground Truth')
  # blue_cross = mlines.Line2D([], [], color='dodgerblue', marker='P', linestyle='None',
  #                         markersize=12, label='Pred Mean Field')
  # red_cross = mlines.Line2D([], [], color='orangered', marker='P', linestyle='None',
  #                         markersize=12, label='True Mean Field')
  legend_handle = []
  if include_true:
    legend_handle.append(red_dot)
    # legend_handle.append(red_cross)
  if include_pred:
    legend_handle.append(blue_dot)
    # legend_handle.append(blue_cross)
  plt.legend(handles=legend_handle)

  pred_len, AGENT_COUNT,_ = pred_traj.shape
  trailing_buf_size = 25
  total_frame = pred_len if total_frame > pred_len else total_frame
  print("Number of steps: {0}\nAnimating {1} steps\nNumer of agents: {1}".format(pred_len, total_frame,AGENT_COUNT))

  x = pred_traj[:,:,0]
  y = pred_traj[:,:,1]
  z = pred_traj[:,:,2]
  x_gt = true_traj[:,:,0] #gt stand for ground truth
  y_gt = true_traj[:,:,1]
  z_gt = true_traj[:,:,2]

  x_v = pred_vel[:,:,0]
  y_v = pred_vel[:,:,1]
  z_v = pred_vel[:,:,2]
  x_v_gt = true_vel[:,:,0]
  y_v_gt = true_vel[:,:,1]
  z_v_gt = true_vel[:,:,2]

  mean_x = np.sum(x,axis=1)/AGENT_COUNT
  mean_y = np.sum(y,axis=1)/AGENT_COUNT
  mean_z = np.sum(z,axis=1)/AGENT_COUNT
  mean_x_gt = np.sum(x_gt,axis=1)/AGENT_COUNT
  mean_y_gt = np.sum(y_gt,axis=1)/AGENT_COUNT
  mean_z_gt = np.sum(z_gt,axis=1)/AGENT_COUNT

  #x_min = min(np.amin(x)-0.5, np.amin(x_gt)-0.5)
  #x_max = max(np.amax(x)+0.5, np.amax(x_gt)+0.5)
  #y_min = min(np.amin(y)-0.5, np.amin(y_gt)-0.5)
  #y_max = max(np.amax(y)+0.5, np.amax(y_gt)+0.5)
  #z_min = min(np.amin(z)-0.5, np.amin(z_gt)-0.5)
  #z_max = max(np.amax(z)+0.5, np.amax(z_gt)+0.5)
  limit = box_size + 0.2
  x_min = -limit
  x_max = limit
  y_min = -limit
  y_max = limit
  z_min = -limit
  z_max = limit


  fig = plt.figure()
  fig.set_size_inches(8, 8)
  ax = p3.Axes3D(fig)
  ax.set_xlim3d(x_min, x_max)
  ax.set_ylim3d(y_min, y_max)
  ax.set_zlim3d(z_min, z_max)
  
  dir_len = 0.5 # length for orientation tick

  # comparing pred and true agents
  if Compare:
    gg = [Line2D([x_gt[0][0], x[0][0]],[y_gt[0][0], y[0][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      gg[j], = ax.plot([x_gt[0][j], x[0][j]],[y_gt[0][j], y[0][j]], [z_gt[0][j], z[0][j]], c='mediumspringgreen',marker='.')
  
  # plotting the trailing paths
  if Trailing and include_true: 
    trail_t = [Line2D([x_gt[0][0]],[y_gt[0][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      trail_t[j], = ax.plot([x_gt[0,j]],
                            [y_gt[0,j]], 
                            z_gt[0,j], c='orangered',marker=None,alpha=0.3)
  
  if Trailing and include_pred: 
    trail_p = [Line2D([x[0][0]],[y[0][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      trail_p[j], = ax.plot([x[0,j]],
                            [y[0,j]], 
                            z[0,j], c='dodgerblue',marker=None,alpha=0.3)
  
  # Communication range visualization, only for prediction
  if Comm and include_pred:
    tcomm = [Line2D([x[0][0]],[y[0][0]]) for i in range(int((AGENT_COUNT**2-AGENT_COUNT)/2))]
    comm_counter = 0
    xlst, ylst, zlst = x[0], y[0], z[0]
    for i in range(AGENT_COUNT):
      for j in range(i+1, AGENT_COUNT):
        dist = np.sqrt((xlst[i]-xlst[j])**2 + (ylst[i]-ylst[j])**2 + (zlst[i]-zlst[j])**2)
        if dist <= percept_range:
          tcomm[comm_counter], = ax.plot([xlst[i],xlst[j]],
                              [ylst[i],ylst[j]],
                              [zlst[i],zlst[j]], c='cyan',alpha=0.3)
        else:
          tcomm[comm_counter], = ax.plot([xlst[i],xlst[i]],
                              [ylst[i],ylst[i]],
                              [zlst[i],zlst[i]], c='cyan',alpha=0.3)
        comm_counter+=1


  # plotting predicted trajectories
  if include_pred: 
    p, = ax.plot(x[0],y[0],z[0], c='dodgerblue',marker='o',markersize=5, linestyle='None')
    # p_mean, = ax.plot([mean_x[0]],[mean_y[0]],mean_z[0],c='dodgerblue',marker='P', markersize=10, linestyle='None')
    pred_vel_line = [Line2D([x[0][0], dir_len*x_v[0][0]+x[0][0]],
                            [y[0][0], dir_len*y_v[0][0]+y[0][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      pred_vel_line[j], = ax.plot([x[0][j], dir_len*x_v[0][j]+x[0][j]],
                                  [y[0][j], dir_len*y_v[0][j]+y[0][j]],
                                  [z[0][j], dir_len*z_v[0][j]+z[0][j]], 
                                  c='dodgerblue')
  
  # plotting true trajectories
  if include_true:
    t, = ax.plot(x_gt[0],y_gt[0],z_gt[0], c='orangered',marker='o', markersize=5, linestyle='None')
    # t_mean, = ax.plot([mean_x_gt[0]],[mean_y_gt[0]],mean_z_gt[0], c='orangered',marker='P', markersize=10, linestyle='None')
    true_vel_line = [Line2D([x_gt[0][0], dir_len*x_v_gt[0][0]+x_gt[0][0]],
                            [y_gt[0][0], dir_len*y_v_gt[0][0]+y_gt[0][0]]) for i in range(AGENT_COUNT)]
    
    for j in range(AGENT_COUNT):
      true_vel_line[j], = ax.plot([x_gt[0][j], dir_len*x_v_gt[0][j]+x_gt[0][j]],
                                  [y_gt[0][j], dir_len*y_v_gt[0][j]+y_gt[0][j]],
                                  [z_gt[0][j], dir_len*z_v_gt[0][j]+z_gt[0][j]], 
                                  c='orangered')

  def animate(i):
      if i%20==0:
        print("Animating frame {0}/{1}".format(i, total_frame))

      ax.view_init(elev=10., azim=i/2)
      plt.legend(handles=legend_handle)
      if Compare:
        for j in range(AGENT_COUNT):
          gg[j].set_data([x_gt[i][j], x[i][j]],[y_gt[i][j], y[i][j]])
          gg[j].set_3d_properties([z_gt[i][j], z[i][j]])
      
      if Comm and include_pred:
        xlst, ylst, zlst = x[i], y[i], z[i]
        comm_counter=0
        for j in range(AGENT_COUNT):
          for k in range(j+1, AGENT_COUNT):
            dist = np.sqrt((xlst[j]-xlst[k])**2 + (ylst[j]-ylst[k])**2 + (zlst[j]-zlst[k])**2)
            if dist <= percept_range:
              tcomm[comm_counter].set_data([xlst[j], xlst[k]],[ylst[j], ylst[k]])
              tcomm[comm_counter].set_3d_properties([zlst[j], zlst[k]])
            else:
              tcomm[comm_counter].set_data([xlst[j], xlst[j]],[ylst[j], ylst[j]])
              tcomm[comm_counter].set_3d_properties([zlst[j], zlst[j]])
            comm_counter+=1
      
      trail_offset = trailing_buf_size if i>trailing_buf_size else i
      if Trailing and include_pred:
        for j in range(AGENT_COUNT):
            trail_p[j].set_data(x[i-trail_offset:i,j], y[i-trail_offset:i,j])
            trail_p[j].set_3d_properties(z[i-trail_offset:i,j])

      if Trailing and include_true:
        for j in range(AGENT_COUNT):
            trail_t[j].set_data(x_gt[i-trail_offset:i,j], y_gt[i-trail_offset:i,j])
            trail_t[j].set_3d_properties(z_gt[i-trail_offset:i,j])

      if include_pred:
        p.set_data(x[i],y[i])
        p.set_3d_properties(z[i])
        # p_mean.set_data(mean_x[i],mean_y[i])
        # p_mean.set_3d_properties(mean_z[i])
        for j in range(AGENT_COUNT):
          pred_vel_line[j].set_data([x[i][j], dir_len*x_v[i][j]+x[i][j]],
                                    [y[i][j], dir_len*y_v[i][j]+y[i][j]])
          pred_vel_line[j].set_3d_properties([z[i][j], dir_len*z_v[i][j]+z[i][j]])

      if include_true:
        t.set_data(x_gt[i],y_gt[i])
        t.set_3d_properties(z_gt[i])
        # t_mean.set_data(mean_x_gt[i],mean_y_gt[i])
        # t_mean.set_3d_properties(mean_z_gt[i])
        for j in range(AGENT_COUNT):
          true_vel_line[j].set_data([x_gt[i][j], dir_len*x_v_gt[i][j]+x_gt[i][j]],
                                    [y_gt[i][j], dir_len*y_v_gt[i][j]+y_gt[i][j]])
          true_vel_line[j].set_3d_properties([z_gt[i][j], dir_len*z_v_gt[i][j]+z_gt[i][j]])

      if include_pred and include_true:
        return p, t,
      elif include_pred:
        return p,
      else:
        return t,

  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, frames=total_frame, interval=24, blit=True)
  if save:
    anim.save(save_name, writer=animation.FFMpegWriter(fps=24))
  return anim


def plot_snapshot_2D(true_traj, true_vel, pred_traj, pred_vel, save_name, percept_range,
                  include_true=True, include_pred=False, Compare=False, Trailing=False, Comm=False, save=0, tn=0):
    def scale_marker_size(range):
        return range * 65

    import mpl_toolkits.mplot3d.axes3d as p3
    mpl.rcParams.update({'font.size': 22})

    mpl.rcParams['animation.embed_limit'] = 2 ** 128
    # Making legends
    blue_dot = mlines.Line2D([], [], color='dodgerblue', marker='.', linestyle='None',
                             markersize=15, label='Prediction')
    red_dot = mlines.Line2D([], [], color='orangered', marker='.', linestyle='None',
                            markersize=15, label='Ground Truth')

    legend_handle = []
    if include_true:
        legend_handle.append(red_dot)
    if include_pred:
        legend_handle.append(blue_dot)

    # pred_traj = np.squeeze(pred_traj)
    # true_traj = np.squeeze(true_traj)
    # pred_vel = np.squeeze(pred_vel)
    # true_vel = np.squeeze(true_vel)

    pred_len, AGENT_COUNT, _ = pred_traj.shape
    dir_len=0.3
    trailing_buf_size = 25

    x = pred_traj[:, :, 0]
    y = pred_traj[:, :, 1]
    # z = pred_traj[:, :, 2]
    x_gt = true_traj[:, :, 0]  # gt stand for ground truth
    y_gt = true_traj[:, :, 1]
    # z_gt = true_traj[:, :, 2]

    x_v = pred_vel[:, :, 0]
    y_v = pred_vel[:, :, 1]
    # z_v = pred_vel[:, :, 2]
    x_v_gt = true_vel[:, :, 0]
    y_v_gt = true_vel[:, :, 1]
    # z_v_gt = true_vel[:, :, 2]

    # x_min = -5.2
    # x_max = 5.2
    # y_min = -5.2
    # y_max = 5.2
    x_min = np.min([np.min(x), np.min(x_gt)])-3
    x_max = np.max([np.max(x), np.max(x_gt)])+3
    y_min = np.min([np.min(y), np.min(y_gt)])-3
    y_max = np.max([np.max(y), np.max(y_gt)])+3
    # z_min = -5.2
    # z_max = 5.2

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    # set agent's position and velocity in first frame.
    if include_pred:
        p, = ax.plot(x[0], y[0], c='dodgerblue', marker='o', markersize=10, linestyle='None')
        # p_mean, = ax.plot([mean_x[0]], [mean_y[0]], mean_z[0], c='dodgerblue', marker='P', markersize=10,
        #                   linestyle='None')
        pred_vel_line = [Line2D([x[0][0], dir_len * x_v[0][0] + x[0][0]],
                                [y[0][0], dir_len * y_v[0][0] + y[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            pred_vel_line[j], = ax.plot([x[0][j], dir_len * x_v[0][j] + x[0][j]],
                                        [y[0][j], dir_len * y_v[0][j] + y[0][j]],
                                        c='dodgerblue')

    # plotting true trajectories
    if include_true:
        t, = ax.plot(x_gt[0], y_gt[0], c='orangered', marker='o', markersize=10, linestyle='None')
        # t_mean, = ax.plot([mean_x_gt[0]], [mean_y_gt[0]], mean_z_gt[0], c='orangered', marker='P', markersize=10,
        #                   linestyle='None')
        true_vel_line = [Line2D([x_gt[0][0], dir_len * x_v_gt[0][0] + x_gt[0][0]],
                                [y_gt[0][0], dir_len * y_v_gt[0][0] + y_gt[0][0]]) for i in range(AGENT_COUNT)]

        for j in range(AGENT_COUNT):
            true_vel_line[j], = ax.plot([x_gt[0][j], dir_len * x_v_gt[0][j] + x_gt[0][j]],
                                        [y_gt[0][j], dir_len * y_v_gt[0][j] + y_gt[0][j]],
                                        c='orangered')

    # plotting the trailing paths
    if Trailing and include_true:
        trail_t = [Line2D([x_gt[0][0]], [y_gt[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            trail_t[j], = ax.plot([x_gt[0, j]],
                                  [y_gt[0, j]],
                                  c='orangered', marker=None, alpha=0.3)

    if Trailing and include_pred:
        trail_p = [Line2D([x[0][0]], [y[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            trail_p[j], = ax.plot([x[0, j]],
                                  [y[0, j]],
                                  c='dodgerblue', marker=None, alpha=0.3)

    plt.legend(handles=legend_handle, loc='upper right')
    plt.savefig(save_name + '.png', format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)



def plot_snapshot(true_traj, true_vel, pred_traj, pred_vel, save_name, percept_range,
                   include_true=True, include_pred=False, Compare=False, Trailing=False, Comm=False, save=0, tn=0):
  def scale_marker_size(range):
    return range*65
  
  import mpl_toolkits.mplot3d.axes3d as p3
  mpl.rcParams.update({'font.size': 22})

  mpl.rcParams['animation.embed_limit'] = 2**128
  # Making legends
  blue_dot = mlines.Line2D([], [], color='dodgerblue', marker='.', linestyle='None',
                            markersize=15, label='Prediction')
  red_dot = mlines.Line2D([], [], color='orangered', marker='.', linestyle='None',
                            markersize=15, label='Ground Truth')
  
  legend_handle = []
  if include_true:
    legend_handle.append(red_dot)
  if include_pred:
    legend_handle.append(blue_dot)

  # pred_traj = np.squeeze(pred_traj)
  # true_traj = np.squeeze(true_traj)
  # pred_vel = np.squeeze(pred_vel)
  # true_vel = np.squeeze(true_vel)

  pred_len, AGENT_COUNT,_ = pred_traj.shape
  trailing_buf_size = 25

  x = pred_traj[:,:,0]
  y = pred_traj[:,:,1]
  z = pred_traj[:,:,2]
  x_gt = true_traj[:,:,0] #gt stand for ground truth
  y_gt = true_traj[:,:,1]
  z_gt = true_traj[:,:,2]

  x_v = pred_vel[:,:,0]
  y_v = pred_vel[:,:,1]
  z_v = pred_vel[:,:,2]
  x_v_gt = true_vel[:,:,0]
  y_v_gt = true_vel[:,:,1]
  z_v_gt = true_vel[:,:,2]

  x_min = -5.2
  x_max = 5.2
  y_min = -5.2
  y_max = 5.2
  z_min = -5.2
  z_max = 5.2


  fig = plt.figure()
  fig.set_size_inches(8, 8)
  ax = p3.Axes3D(fig)
  ax.set_xlim3d(x_min, x_max)
  ax.set_ylim3d(y_min, y_max)
  ax.set_zlim3d(z_min, z_max)
  
  dir_len = 1 # length for orientation tick

  # comparing pred and true agents
  if Compare:
    gg = [Line2D([x_gt[tn][0], x[tn][0]],[y_gt[tn][0], y[tn][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      gg[j], = ax.plot([x_gt[tn][j], x[tn][j]],[y_gt[tn][j], y[tn][j]], [z_gt[tn][j], z[tn][j]], c='mediumspringgreen',marker='.')
  
  # plotting the trailing paths
  if Trailing and include_true: 
    trail_t = [Line2D([x_gt[tn][0]],[y_gt[tn][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      trail_t[j], = ax.plot([x_gt[tn,j]],
                            [y_gt[tn,j]], 
                            z_gt[tn,j], c='orangered',marker=None,alpha=0.3)
  
  if Trailing and include_pred: 
    trail_p = [Line2D([x[tn][0]],[y[tn][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      trail_p[j], = ax.plot([x[tn,j]],
                            [y[tn,j]], 
                            z[tn,j], c='dodgerblue',marker=None,alpha=0.3)
  
  # Communication range visualization, only for prediction
  if Comm and include_pred:
    tcomm = [Line2D([x[tn][0]],[y[tn][0]]) for i in range(int((AGENT_COUNT**2-AGENT_COUNT)/2))]
    comm_counter = 0
    xlst, ylst, zlst = x[tn], y[tn], z[tn]
    for i in range(AGENT_COUNT):
      for j in range(i+1, AGENT_COUNT):
        dist = np.sqrt((xlst[i]-xlst[j])**2 + (ylst[i]-ylst[j])**2 + (zlst[i]-zlst[j])**2)
        if dist <= percept_range:
          tcomm[comm_counter], = ax.plot([xlst[i],xlst[j]],
                              [ylst[i],ylst[j]],
                              [zlst[i],zlst[j]], c='cyan',alpha=0.3)
        else:
          tcomm[comm_counter], = ax.plot([xlst[i],xlst[i]],
                              [ylst[i],ylst[i]],
                              [zlst[i],zlst[i]], c='cyan',alpha=0.3)
        comm_counter+=1


  # plotting predicted trajectories
  if include_pred: 
    p, = ax.plot(x[tn],y[tn],z[tn], c='dodgerblue',marker='o',markersize=10, linestyle='None')
    pred_vel_line = [Line2D([x[tn][0], dir_len*x_v[tn][0]+x[tn][0]],
                            [y[tn][0], dir_len*y_v[tn][0]+y[tn][0]]) for i in range(AGENT_COUNT)]
    for j in range(AGENT_COUNT):
      pred_vel_line[j], = ax.plot([x[tn][j], dir_len*x_v[tn][j]+x[tn][j]],
                                  [y[tn][j], dir_len*y_v[tn][j]+y[tn][j]],
                                  [z[tn][j], dir_len*z_v[tn][j]+z[tn][j]], 
                                  c='dodgerblue', linewidth=2)
  
  # plotting true trajectories
  if include_true:
    t, = ax.plot(x_gt[tn],y_gt[tn],z_gt[tn], c='orangered',marker='o', markersize=10, linestyle='None')
    print(x_gt.shape, x_v_gt.shape)
    true_vel_line = [Line2D([x_gt[tn][0], dir_len*x_v_gt[tn][0]+x_gt[tn][0]],
                            [y_gt[tn][0], dir_len*y_v_gt[tn][0]+y_gt[tn][0]]) for i in range(AGENT_COUNT)]
    
    for j in range(AGENT_COUNT):
      true_vel_line[j], = ax.plot([x_gt[tn][j], dir_len*x_v_gt[tn][j]+x_gt[tn][j]],
                                  [y_gt[tn][j], dir_len*y_v_gt[tn][j]+y_gt[tn][j]],
                                  [z_gt[tn][j], dir_len*z_v_gt[tn][j]+z_gt[tn][j]], 
                                  c='orangered', linewidth=2)
  
  plt.legend(handles=legend_handle)
  plt.savefig(save_name+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)



def plot_2D(position, obs_time=None, save_name='./test.jpg'):
    n_agent = position.shape[1]
    fig = plt.figure()
    # use position data to plot the 3D trajectory
    for obs_index in range(n_agent):
        if obs_time != None:
            x = position[:obs_time, obs_index, 0]
            y = position[:obs_time, obs_index, 1]
        else:
            x = position[:, obs_index, 0]
            y = position[:, obs_index, 1]
        plt.plot(x, y, label='motion' + str(obs_index))
    plt.legend()
    plt.title('2D Trajectory')
    plt.savefig(save_name)
    plt.show()

def make_animation_2D(true_traj, true_vel, pred_traj, pred_vel,
                      total_frame, save_name,
                      include_true=True, include_pred=False, Trailing=True,
                      save=1):
    import mpl_toolkits.mplot3d.axes3d as p3
    mpl.rcParams.update({'font.size': 22})

    mpl.rcParams['animation.embed_limit'] = 2 ** 128
    # Making legends
    blue_dot = mlines.Line2D([], [], color='dodgerblue', marker='.', linestyle='None',
                             markersize=15, label='Prediction')
    red_dot = mlines.Line2D([], [], color='orangered', marker='.', linestyle='None',
                            markersize=15, label='Ground Truth')

    legend_handle = []
    if include_true:
        legend_handle.append(red_dot)
    if include_pred:
        legend_handle.append(blue_dot)


    # animation settings
    t, AGENT_COUNT, _ = true_traj.shape
    trailing_buf_size = 100
    dir_len = 0.5  # length for orientation tick
    # load coordination
    x = pred_traj[:, :, 0]
    y = pred_traj[:, :, 1]
    x_gt = true_traj[:, :, 0]  # gt stand for ground truth
    y_gt = true_traj[:, :, 1]

    x_v = pred_vel[:, :, 0]
    y_v = pred_vel[:, :, 1]
    x_v_gt = true_vel[:, :, 0]
    y_v_gt = true_vel[:, :, 1]


    ## set up the initial frame
    # set frame scale
    x_min = np.min([np.min(x), np.min(x_gt)])-3
    x_max = np.max([np.max(x), np.max(x_gt)])+3
    y_min = np.min([np.min(y), np.min(y_gt)])-3
    y_max = np.max([np.max(y), np.max(y_gt)])+3
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.legend(handles=legend_handle, loc='upper right')
    # set agent's position and velocity in first frame.
    if include_pred:
        p, = ax.plot(x[0], y[0], c='dodgerblue', marker='o', markersize=5, linestyle='None')
        # p_mean, = ax.plot([mean_x[0]], [mean_y[0]], mean_z[0], c='dodgerblue', marker='P', markersize=10,
        #                   linestyle='None')
        pred_vel_line = [Line2D([x[0][0], dir_len * x_v[0][0] + x[0][0]],
                                [y[0][0], dir_len * y_v[0][0] + y[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            pred_vel_line[j], = ax.plot([x[0][j], dir_len * x_v[0][j] + x[0][j]],
                                        [y[0][j], dir_len * y_v[0][j] + y[0][j]],
                                        c='dodgerblue')

    # plotting true trajectories
    if include_true:
        t, = ax.plot(x_gt[0], y_gt[0], c='orangered', marker='o', markersize=5, linestyle='None')
        # t_mean, = ax.plot([mean_x_gt[0]], [mean_y_gt[0]], mean_z_gt[0], c='orangered', marker='P', markersize=10,
        #                   linestyle='None')
        true_vel_line = [Line2D([x_gt[0][0], dir_len * x_v_gt[0][0] + x_gt[0][0]],
                                [y_gt[0][0], dir_len * y_v_gt[0][0] + y_gt[0][0]]) for i in range(AGENT_COUNT)]

        for j in range(AGENT_COUNT):
            true_vel_line[j], = ax.plot([x_gt[0][j], dir_len * x_v_gt[0][j] + x_gt[0][j]],
                                        [y_gt[0][j], dir_len * y_v_gt[0][j] + y_gt[0][j]],
                                        c='orangered')

    # plotting the trailing paths
    if Trailing and include_true:
        trail_t = [Line2D([x_gt[0][0]], [y_gt[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            trail_t[j], = ax.plot([x_gt[0, j]],
                                  [y_gt[0, j]],
                                  c='orangered', marker=None, alpha=0.3)

    if Trailing and include_pred:
        trail_p = [Line2D([x[0][0]], [y[0][0]]) for i in range(AGENT_COUNT)]
        for j in range(AGENT_COUNT):
            trail_p[j], = ax.plot([x[0, j]],
                                  [y[0, j]],
                                  c='dodgerblue', marker=None, alpha=0.3)

    def animate(i):
        i_gt = min(i, 1994)
        if i % 20 == 0:
            print("Animating frame {0}/{1}".format(i, total_frame))

        if include_pred:
            p.set_data(x[i], y[i])
            # p.set_3d_properties(z[i])
            # p_mean.set_data(mean_x[i], mean_y[i])
            # p_mean.set_3d_properties(mean_z[i])
            for j in range(AGENT_COUNT):
                pred_vel_line[j].set_data([x[i][j], dir_len * x_v[i][j] + x[i][j]],
                                          [y[i][j], dir_len * y_v[i][j] + y[i][j]])
                # pred_vel_line[j].set_3d_properties([z[i][j], dir_len * z_v[i][j] + z[i][j]])

        if include_true:
            t.set_data(x_gt[i_gt], y_gt[i_gt])
            # t.set_3d_properties(z_gt[i])
            # t_mean.set_data(mean_x_gt[i], mean_y_gt[i])
            # t_mean.set_3d_properties(mean_z_gt[i])
            for j in range(AGENT_COUNT):
                true_vel_line[j].set_data([x_gt[i_gt][j], dir_len * x_v_gt[i_gt][j] + x_gt[i_gt][j]],
                                          [y_gt[i_gt][j], dir_len * y_v_gt[i_gt][j] + y_gt[i_gt][j]])
                # true_vel_line[j].set_3d_properties([z_gt[i][j], dir_len * z_v_gt[i][j] + z_gt[i][j]])

        trail_offset = trailing_buf_size if i > trailing_buf_size else i
        if Trailing and include_pred:
            for j in range(AGENT_COUNT):
                trail_p[j].set_data(x[i - trail_offset:i, j], y[i - trail_offset:i, j])

        if Trailing and include_true:
            for j in range(AGENT_COUNT):
                trail_t[j].set_data(x_gt[i_gt - trail_offset:i_gt, j], y_gt[i_gt - trail_offset:i_gt, j])


        if include_pred and include_true:
            return p, t,
        elif include_pred:
            return p,
        else:
            return t,


    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=total_frame, interval=24, blit=True)
    if save:
        anim.save(save_name, writer=animation.FFMpegWriter(fps=90))
    return anim
