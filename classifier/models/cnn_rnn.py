import numpy as np
import torch
from torch import nn


class CNNRNN(nn.Module):
    """
    Maybe further ideas:
    https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210
    https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145

    For this to train properly needs >= 100ep
    Other lr does not help.
    LSTM and RNN not working, but GRU
    -> even with manual init not working

    Most GPU RAM is required by the CNN (because this is stored #nr_layers times)
    """

    def __init__(self, crop_size, pretrained=False, num_classes=2, in_chans=1, dropout=0.4, hparams=None):
        super(CNNRNN, self).__init__()
        import timm

        self.hparams = hparams
        
        # from efficientnet_pytorch import EfficientNet
        # final_nr_filt = 128
        # self.cnn = EfficientNet.from_pretrained("efficientnet-b0", in_channels=in_chans, num_classes=num_classes)
        # self.cnn._fc = nn.Identity()  # "deactivate" classifier; 1280 output features
        # self.lstm = nn.LSTM(input_size=final_nr_filt*10,
        #                     hidden_size=final_nr_filt//4,
        #                     num_layers=1,
        #                     batch_first=True, 
        #                     bidirectional=False)
        # self.classifier = nn.Linear(final_nr_filt//4, num_classes)


        # Rough heuristic: 
        # shape before avp_pool: s = int((input_size / 2**nr_maxpool) - 5)
        # s should be slightly bigger than avg_pool_size
        # avg_pool_size = [8, 8]
        avg_pool_size = [4, 4]

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_chans, 8, 3),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(8, 16, 3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(16, 32, 3),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),

        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),   # For input [290, 350] -> [14, 18]

        #     nn.AdaptiveAvgPool2d(avg_pool_size),
        #     nn.Flatten()  # flatten all but bs dim
        # )
        # final_nr_filt = 128

        # size of features: 
        #  global_pool=''   -> [8, 1280, 25, 38]
        #  global_pool='avg' -> [8, 1280]
        # num_classes=0 removes the classification head
        self.cnn = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained,
                                     num_classes=0, in_chans=in_chans,
                                     drop_rate=dropout, global_pool='avg')
        avg_pool_size = [1, 1]
        final_nr_filt = 1280

        
        final_nr_filt_fc = final_nr_filt
        # For ICBbig gender prediction bidirectional works worse (more instable + slightly worse)
        bidirectional = False
        if bidirectional:
            final_nr_filt_fc *= 2

        seq_model = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[self.hparams.tf_seq_model]
        self.lstm = seq_model(input_size=final_nr_filt*np.prod(avg_pool_size),
                              hidden_size=int(final_nr_filt * self.hparams.tf_embed_multiplier),   # hidden to be nr_filt/2 best for ICBbig gender prediction
                              num_layers=self.hparams.tf_nr_layers,
                              batch_first=True, 
                              bidirectional=bidirectional)   # bidirectional: lstm output shape: [bs, z, nr_hidden*2]
        self.classifier = nn.Linear(int(final_nr_filt * self.hparams.tf_embed_multiplier), num_classes)

        # init right here (no improvement)
        # for layer_p in self.lstm._all_weights:
        #     for p in layer_p:
        #         if 'weight' in p:
        #             # print(p, a.__getattr__(p))
        #             nn.init.normal_(self.lstm.__getattr__(p), 0.0, 0.02)
        #             # nn.init.xavier_uniform_(self.lstm.__getattr__(p))
        #             # print(p, a.__getattr__(p))

        # init in extra function (no improvement)
        # self.init_weights()

        # Without LSTM
        # self.classifier = nn.Linear(final_nr_filt*np.prod(avg_pool_size)*40, num_classes)

        # self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=3,
        #                                   num_decoder_layers=3, dim_feedforward=512, dropout=0.0,
        #                                   activation='relu', custom_encoder=None, custom_decoder=None,
        #                                   layer_norm_eps=1e-05, batch_first=True)

        # self.classifier = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(32, num_classes),
        # )

    # def init_weights(self):
    #     for m in self.modules():
    #         if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
    #             for name, param in m.named_parameters():
    #                 if 'weight_ih' in name:
    #                     torch.nn.init.xavier_uniform_(param.data)
    #                 elif 'weight_hh' in name:
    #                     torch.nn.init.orthogonal_(param.data)
    #                 elif 'bias' in name:
    #                     param.data.fill_(0)

    def forward(self, x):
        bs, C, X, Y, Z = x.size()  # input 3D: [bs, channel, x, y, z]
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(bs*Z, C, X, Y)  # transform to series of 2D: [bs*z, channel, x, y]
        x = self.cnn(x)  # [bs*z, embedding]

        x = x.view(bs, Z, -1)  # change to series: [bs, z, embedding]
        # print(f"x.shape 4: {x.shape}")
        # x, (h_n, h_c) = self.lstm(x)  # x: [bs, z, nr_hidden]  (lstm output), h_n: (nr_layers, bs, nr_hidden)  (final hidden state)
        
        # initialize the hidden state (maybe slightly better)
        # hidden_init = torch.randn(1, bs, 128).cuda()  # [nr_layers, bs, nr_hidden]
        # x, h_n = self.lstm(x, hidden_init)
        
        x, h_n = self.lstm(x)  # for GRU
        # print(f"x.shape 5: {x.shape}")
        # x = self.classifier(x[:, -1, :])  # only use last timepoint
        x = self.classifier(x[:, :, :]).mean(dim=1)  # for ICBbig gender prediction mean works slightly better
        # x = self.classifier(h_n[-1])

        # Without any recurrent layer
        # x = x.view(bs, -1)
        # x = self.classifier(x)
        
        return x  # output: [bs, classes]
