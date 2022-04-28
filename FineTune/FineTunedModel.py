import torch
from torch import nn
import copy
from encoder_decoder import Encoder,EncoderLayer,MultiHeadAttention,PositionwiseFeedForward
from utils import Set2Set,Embedding,PositionalEncoding,Regression_fun
H = 16 #16
P = 0.3
N = 5
D_MODEL = 512 #512
NUM_ATOM = 96 # 0：PAD_NUM  95: masked_token for pretraining
D_FF = 2048 #2048
PAD_NUM = 0


class FineTunedModel(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, h=H, dropout=P, num_layer=N, num_atom=NUM_ATOM, pad_num=PAD_NUM):
        super(FineTunedModel, self).__init__()
        self.Embedding = Embedding(num_atom, d_model)
        self.PositionalEncoding = PositionalEncoding(dropout, d_model, pad_num)
        self.atten = MultiHeadAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(self.atten), copy.deepcopy(self.ff), dropout),
                               num_layer)
        self.set2set = Set2Set(d_model, T=10)
        self.regression_supercon = Regression_fun(2 * d_model)

    def forward(self, atom_index_data, coords_data, mask, mask_for_set2set):
        # atom_index_data: batch_size,max_seq_len
        mask = mask.unsqueeze(1)  # batch_size,1,max_seq
        atom_feature = self.Embedding(atom_index_data)  # batch_size,max_seq_len,d_model
        atom_feature = self.PositionalEncoding(atom_feature, coords_data)
        atom_feature = self.encoder(atom_feature, mask)
        mol_feature = self.set2set(atom_feature, mask_for_set2set)
        return self.regression_supercon(mol_feature).squeeze()


if __name__ == "__main__":
    fineTuneModel = FineTunedModel()
    for param in fineTuneModel.parameters():
        if param.dim() > 1:
            nn.init.kaiming_normal_(param)
    fineTunedModel_dic = fineTuneModel.state_dict()
    pretrained_dic = torch.load("model/pretrained.pt")
    common_param_dic = {k:v for k,v in pretrained_dic.items() if k in fineTunedModel_dic}
    fineTunedModel_dic.update(common_param_dic)
    fineTuneModel.load_state_dict(fineTunedModel_dic)
    print(len(common_param_dic))
    print(pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].device, #cuda:0
          fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"].device) #cpu
    print(fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"] == pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].cpu())
    print(pretrained_dic["encoder.layers.18.self_atten.linears.0.weight"].device,fineTuneModel.state_dict()["encoder.layers.18.self_atten.linears.0.weight"].device)