import numpy as np
import torch
from torch import nn
# from transformers.modeling_distilbert import Transformer as _Transformer


class CNNTransformer(nn.Module):
    """
    https://github.com/i-pan/kaggle-rsna-pe/blob/main/src/beehive/models/sequence.py

    https://discuss.pytorch.org/t/nn-transformerencoder-for-classification/83021/3

    DistilBertModel: ep100_lr1e-5   (for DistilBertModel 50ep not enough)

    Pytorch Transformer working better (ep50_lr5e-6)

    Good for ICBbig gender: ep50, lr1e-4, tf_nr_layers 1, tf_nr_heads 4, tf_embed_multiplier 1.0
    """

    def __init__(self, crop_size, pretrained=False, num_classes=2, in_chans=1, dropout=0.4, hparams=None):
        super(CNNTransformer, self).__init__()
        import timm

        self.hparams = hparams

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
        #     # nn.Conv2d(64, 128, 3),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),

        #     nn.AdaptiveAvgPool2d(avg_pool_size),
        #     nn.Flatten()  # flatten all but bs dim
        # )
        # # final_nr_filt = 128
        # final_nr_filt = 64

        self.cnn = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained,
                                     num_classes=0, in_chans=in_chans,
                                     drop_rate=dropout, global_pool='avg')
        avg_pool_size = [1, 1]
        final_nr_filt = 1280


        embedding_dim = final_nr_filt*np.prod(avg_pool_size)  # 
        self.n_layers = self.hparams.tf_nr_layers  # 2
        self.n_heads = self.hparams.tf_nr_heads  # 4

        # from transformers import DistilBertModel, DistilBertConfig
        # config = DistilBertConfig(
        #     vocab_size=1,  # 30522
        #     max_position_embeddings=40,
        #     sinusoidal_pos_embds=False,
        #     n_layers=self.n_layers,
        #     n_heads=self.n_heads,
        #     dim=embedding_dim,
        #     hidden_dim=2*embedding_dim,  #  4*
        #     dropout=0.2, 
        #     attention_dropout=0.1,
        #     activation="gelu",
        #     initializer_range=0.02, 
        #     qa_dropout=0.1,
        #     seq_classif_dropout=0.2,
        #     pad_token_id=0)
        # self.transformer = DistilBertModel(config)

        # n_hidden = int(embedding_dim*2)   # doing 1*embedding_dim at leat for 1024 not working
        n_hidden = int(embedding_dim * self.hparams.tf_embed_multiplier)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=self.n_heads,
                                                   dim_feedforward=n_hidden, dropout=dropout, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)


    def forward(self, x):
        bs, C, X, Y, Z = x.size()  # input 3D: [bs, channel, x, y, z]
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(bs*Z, C, X, Y)  # transform to series of 2D: [bs*z, channel, x, y]
        x = self.cnn(x)  # [bs*z, embedding]
        x = x.view(bs, Z, -1)  # change to series: [bs, z, embedding]
        
        # mask = torch.from_numpy(np.ones((bs, x.size(1)))).long().to(x.device)
        # out = self.transformer(inputs_embeds=x, attention_mask=mask, head_mask=torch.ones(self.n_layers, self.n_heads).to(x.device))
        # x = out.last_hidden_state  # [bs, sequence_length, hidden_size]

        x = x.permute(1, 0, 2)  # [bs, z, embedding]  # move seq dim to front
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        # x = x[:, -1, :]  # only last element of sequence
        x = x.mean(dim=1)  # mean over seqeuence; maybe slightly better?

        # print(f"x.shape 5: {x.shape}")
        x = self.classifier(x)  # which element to take from sequence?? First and last works equally bad
        
        return x  # output: [bs, classes]
