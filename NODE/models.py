from NODE.NODE import *


class WaveRiders(ODEF):
    def __init__(self, device):  # quadparams is only a placeholder, not actually used
        super(WaveRiders, self).__init__()
        self.lin1 = nn.Linear(4, 256)
        self.lin2 = nn.Linear(256, 2)
        self.activ = nn.Tanh()
        self.device = device

    def forward(self, z):
        bs, *r = z.size()
        z = z.squeeze(1)
        latlon = z[:, :2]  # 2D state lat lon
        uv_vel = z[:, 2:]  # 2D input, u_vel, v_vel

        x_dot = self.activ(self.lin1(z))
        x_dot = self.lin2(x_dot)

        out = torch.cat([torch.zeros([bs, 2]).to(self.device), x_dot], 1)
        return out.unsqueeze(1)


class WaveRiders3Day(ODEF):
    def __init__(self, device):  # quadparams is only a placeholder, not actually used
        super(WaveRiders3Day, self).__init__()
        self.lin1 = nn.Linear(4, 256)
        self.lin2 = nn.Linear(256, 2)
        self.activ = nn.Tanh()
        self.device = device
        self.time_conv = nn.Conv1d(2, 2, 3)
        # self.time_lin1 = nn.Linear(6, 16)
        # self.time_lin2 = nn.Linear(16, 2)

    def forward(self, z):
        bs, *r = z.size()
        z = z.squeeze(1)
        latlon = z[:, -2:]  # 2D state lat lon
        uv_vel_3day = z[:, :-2]  # 2D input, u_vel, v_vel
        u_vel_3day = uv_vel_3day[:, ::2].unsqueeze(1)
        v_vel_3day = uv_vel_3day[:, 1::2].unsqueeze(1)
        uv_vel_3day_stacked = torch.cat([u_vel_3day, v_vel_3day], 1)
        uv_vel = self.time_conv(uv_vel_3day_stacked).squeeze()
        # uv_vel = self.activ(self.time_lin1(uv_vel_3day))
        # uv_vel = self.time_lin2(uv_vel)

        x_dot = torch.cat([latlon, uv_vel.view(-1,2)], 1)
        x_dot = self.activ(self.lin1(x_dot))
        x_dot = self.lin2(x_dot)

        out = torch.cat([torch.zeros([bs, 6]).to(self.device), x_dot], 1)
        return out.unsqueeze(1)
