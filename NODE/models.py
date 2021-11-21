from NODE.NODE import *


class WaveRiders(ODEF):
    def __init__(self, device):  # quadparams is only a placeholder, not actually used
        super(WaveRiders, self).__init__()
        self.lin1 = nn.Linear(4, 256)
        self.lin2 = nn.Linear(256, 2)
        self.tanh = nn.Tanh()
        self.device = device

    def forward(self, z):
        bs, *r = z.size()
        z = z.squeeze(1)
        latlon = z[:, :2]  # 2D state lat lon
        uv_vel = z[:, 2:]  # 2D input, u_vel, v_vel

        x_dot = self.tanh(self.lin1(z))
        x_dot = self.lin2(x_dot)

        out = torch.cat([torch.zeros([bs, 2]).to(self.device), x_dot], 1)
        return out.unsqueeze(1)