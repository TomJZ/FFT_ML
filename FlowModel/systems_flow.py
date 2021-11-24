import torch
from torch import nn
from NODE.NODE import ODEF
from torch import Tensor
import math

class FlowNN(ODEF):
    """
    Flow Dynamic NN model in the Neural ODE formulation.
    """
    def __init__(self, device):
        super(FlowNN, self).__init__()
        self.conv2d1 = nn.Conv2d(2, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2d2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2d3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(192, 2106)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.l1 = nn.Sequential(self.conv2d1,
                                self.bn1,
                                self.tanh,
                                self.maxpool2d)

        self.l2 = nn.Sequential(self.conv2d2,
                                self.bn2,
                                self.tanh,
                                self.maxpool2d)

        self.l3 = nn.Sequential(self.conv2d3,
                                self.bn3,
                                self.tanh,
                                self.maxpool2d)

        self.l4 = nn.Sequential(self.fc1)
        # self.l5 = nn.Sequential(self.fc2,
        #                         self.tanh)
        # self.l6 = nn.Sequential(self.fc3,
        #                         self.tanh)

        # device setting
        self.device = device

    def forward(self, x_curr):
        '''
        # feature engineer
        t_curr = (torch.arange(x_curr.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x_curr) * torch.ones_like(x_curr[:,:,:,:,0])).unsqueeze(-1)
        # aug x state
        x_curr_aug = torch.cat([x_curr, t_curr], dim=-1)
        x = x_curr_aug.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        '''

        x = x_curr.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        # CNN forward
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # fc forward
        x = x.flatten(start_dim=1, end_dim=3)
        x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        x = x.view(x_curr.shape)
        return x



class TransformerModel(nn.Module):
    def __init__(self, s_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder_2d = PositionalEncoding2D(dropout)

        # down/up sampling layer
        self.down_sample = nn.Upsample([25, 40], mode="bilinear")
        self.up_sample = nn.Upsample([500, 800], mode="bilinear")


        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, activation='gelu',norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.src_embedding_conv = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1),
                                                # nn.BatchNorm2d(16),
                                                # nn.MaxPool2d(2),
                                                # nn.Conv2d(16, 64, 3, padding=1),
                                                # nn.BatchNorm2d(64),
                                                # nn.MaxPool2d(2),
                                                # nn.Conv2d(64, 64, 3, padding=1),
                                                )
        self.src_embedding_fc = nn.Linear(10000, d_model)

        self.tgt_embedding_conv = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1),
                                                # nn.BatchNorm2d(16),
                                                # nn.MaxPool2d(2),
                                                # nn.Conv2d(16, 64, 3, padding=1),
                                                # nn.BatchNorm2d(64),
                                                # nn.MaxPool2d(2),
                                                # nn.Conv2d(64, 64, 3, padding=1),
                                                )

        self.tgt_embedding_fc = nn.Linear(10000, d_model)


        self.relu = nn.ReLU()

        # Transformer decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, activation='gelu',norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 10000)
        self.smooth = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1),
                                    # nn.BatchNorm2d(2),
                                    # nn.Conv2d(2, 2, 3, padding=1),
                                    )

        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.src_embedding_fc.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding_fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, memory_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, s_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, s_dim]
        """
        # size constant
        bs, pred_seq_len, x_grid, y_grid, dim = tgt.size()

        # 2d postition
        src = self.pos_encoder_2d(src)
        tgt = self.pos_encoder_2d(tgt)
        # CNN layers
        src_ = src.flatten(end_dim=1).permute(0,3,1,2)
        tgt_ = tgt.flatten(end_dim=1).permute(0,3,1,2)
        src_cnn = self.src_embedding_conv(src_).permute(0,2,3,1)
        src_cnn = src_cnn.reshape(*src.shape[:2], *src_cnn.shape[1:])
        tgt_cnn = self.tgt_embedding_conv(tgt_).permute(0,2,3,1)
        tgt_cnn = tgt_cnn.reshape(*tgt.shape[:2], *tgt_cnn.shape[1:])
        # temp code
        src_flat = src_cnn.flatten(start_dim=2, end_dim=4)
        tgt_flat = tgt_cnn.flatten(start_dim=2, end_dim=4)

        # src = self.encoder(src) * math.sqrt(self.d_model)
        # embedding for the source and target.
        src_em = self.relu(self.src_embedding_fc(src_flat))
        tgt_em = self.relu(self.tgt_embedding_fc(tgt_flat))

        # '''
        # src encode
        # src_em = self.pos_encoder(src_em)
        memory = self.transformer_encoder(src_em, src_mask)

        # decoder: memory & target
        # tgt_em = self.pos_encoder(tgt_em)
        output = self.transformer_decoder(tgt_em, memory, tgt_mask, memory_mask)
        # '''
        # output = self.transformer(src_em, tgt_em, src_mask, tgt_mask, None,)
        output = self.decoder(output)
        output = output.reshape(tgt.size())
        output = self.smooth(output.flatten(end_dim=1).permute(0,3,1,2)).permute(0,2,3,1).reshape(tgt.size())
        return output


def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src=None, tgt=None, src_len=None, tgt_len=None, DEVICE='cuda:0'):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag for target mask.
    zero mask for source mask
    :param src: [bs, len, dim]
    :param tgt: [bs, len, dim]
    :return:
    """
    if src_len == None or tgt_len == None:
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]

    src_mask = torch.zeros((src_len, src_len), device=DEVICE).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_len, DEVICE)
    # memory_mask = torch.zeros((tgt_len, src_len), device=DEVICE).type(torch.bool)
    memory_mask = None

    return src_mask, tgt_mask, memory_mask

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


class PositionalEncoding2D(nn.Module):
    def __init__(self,
                 dropout: float):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor):
        # torch.meshgrid(torch.arange(token_embedding.size()))
        bs, seq_len, x_grid, y_grid, dim = token_embedding.size()
        xx, yy = torch.meshgrid(torch.linspace(-1, 1, x_grid), torch.linspace(-1, 1, y_grid), indexing='ij')
        coordinate_channel = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], dim=-1).unsqueeze(0).unsqueeze(0).to(token_embedding)
        return self.dropout(token_embedding + coordinate_channel)
