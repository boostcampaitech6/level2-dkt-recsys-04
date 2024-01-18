import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

import pandas as pd


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.cat_cols = args.cat_cols
        self.con_cols = args.con_cols
        
        # [건우] 부모객체(ModelBase)에 hidden_dim, n_layers, n_heads, drop_out, max_seq_len 올려놓음
        self.hidden_dim = args.hidden_dim # 이전 sequence output size
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out
        self.max_seq_len = args.max_seq_len
        
        # [건우] category변수의 unique의 개수를 저장한 것을 불러옴
        self.n_args = [arg for arg in vars(self.args) if arg.startswith('n_')]
        for arg in self.n_args:
            value = getattr(self.args, arg)
            setattr(self, arg, value) # self로 현재 class의 attribute로 불러옴

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = self.hidden_dim, self.hidden_dim // 3

        # [건우] category_feature개수 만큼 nn.Embedding 만듬(interaction만 args에 파싱 후에 만들어졌기 때문에 따로 만듬)
        self.embedding_interaction = nn.Embedding(3, intd) # interaction이란 이전 sequence를 학습하기 위한 column(correct(1(성공), 2(실패)) + padding(0)) 
        for cat_col in self.args.cat_cols:
            n = getattr(self, f'n_{cat_col}') # n = self.n_xx 의 값 
            setattr(self, f'embedding_{cat_col}', nn.Embedding(n + 1, intd)) # self.embedding_xx = nn.Embedding(n + 1, intd)

        # [건우] nn.Linear의 첫 번째 argument 수정
        self.comb_proj = nn.Linear(intd * (len(self.args.cat_cols) +1)+len(self.args.con_cols), hd) # intd가 embedding차원이라 category만 적용

        # Fully connected layer
        self.fc = nn.Linear(hd, 1) # 통과하면 feature차원이 1
    
    def forward(self, data):
        interaction = data["interaction"]
        batch_size = interaction.size(0)

        ####### [건우] Embedding + concat ######  
        # category embeding + concat
        embed_interaction = self.embedding_interaction(interaction.int()) 

        embed_cat_feats = []
        for cat_col in self.args.cat_cols:
            value = data[cat_col]
            embed_cat_feat = getattr(self, f'embedding_{cat_col}')(value.int()) # self.embedding_xxx(xxx.int())
            embed_cat_feats.append(embed_cat_feat)
        embed = torch.cat([embed_interaction,*embed_cat_feats],dim=2) # dim=2는 3차원을 합친다는 의미

        # continious concat
        con_feats = []
        for con_col in self.args.con_cols: 
            value = data[con_col]
            con_feats.append(value.unsqueeze(2))
        embed = torch.cat([embed,*con_feats], dim=2).float()
        ################# [건우] ###############
        
        X = self.comb_proj(embed) # concat후 feature_size=Hidden dimension으로 선형변환
        return X, batch_size # embedding을 concat하고 선형 변환한 값


class LSTM(ModelBase):
    def __init__(self, args): # [건우] : args로 몰았기 때문에 args만 씀
        super().__init__(args) # [건우] : args로 몰았기 때문에 args만 씀
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True # input_size=hidden_size
        ) 

    def forward(self, data):
        # X는 embedding들을 concat한 값
        # super().forward은 부모객체의 forward메소드를 말함
        X, batch_size = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(self,args): # [건우] : args로 몰았기 때문에 args만 씀
        super().__init__(args) # [건우] : args로 몰았기 때문에 args만 씀
        
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, data):
        X, batch_size = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)


        extended_attention_mask = data["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(self,args): # [건우] : args로 몰았기 때문에 args만 씀
        super().__init__(args) # [건우] : args로 몰았기 때문에 args만 씀

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=self.max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, data):
        X, batch_size = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=data["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
