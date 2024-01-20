import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import torch.nn.functional as F
import numpy as np
import math # [승준] positional encoding을 위한 math 패키지, numpy 추가
import re

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
        self.device = args.device
        
        # [건우] category변수의 unique의 개수를 저장한 것을 불러옴
        self.n_args = [arg for arg in vars(self.args) if arg.startswith('n_')]
        for arg in self.n_args:
            value = getattr(self.args, arg)
            setattr(self, arg, value) # self로 현재 class의 attribute로 불러옴

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = self.hidden_dim, self.hidden_dim // len(self.args.cat_cols)

        # [건우] category_feature개수 만큼 nn.Embedding 만듬(interaction만 args에 파싱 후에 만들어졌기 때문에 따로 만듬)
        self.embedding_interaction = nn.Embedding(3, intd) # interaction이란 이전 sequence를 학습하기 위한 column(correct(1(성공), 2(실패)) + padding(0)) 
        for cat_col in self.args.cat_cols:
            n = getattr(self, f'n_{cat_col}') # n = self.n_xx 의 값 
            setattr(self, f'embedding_{cat_col}', nn.Embedding(n + 1, intd)) # self.embedding_xx = nn.Embedding(n + 1, intd)

        # [건우] nn.Linear의 첫 번째 argument 수정
        self.comb_proj = nn.Linear(intd * (len(self.args.cat_cols) +1)+len(self.args.con_cols), hd) # intd가 embedding차원이라 category만 적용
        # [승준] encoder comb_proj 추가
        self.enc_comb_proj = nn.Linear(intd * (len(self.args.cat_cols))+len(self.args.con_cols), hd)
        # Fully connected layer
        self.fc = nn.Linear(hd, 1) # 통과하면 feature차원이 1
    
    def forward(self, data):
        interaction = data["interaction"]
        batch_size = interaction.size(0)
        # [찬우] seq_len 추가
        seq_len = interaction.size(1)

        ####### [건우] Embedding + concat ######  
        # category embeding + concat
        embed_interaction = self.embedding_interaction(interaction.int()) 

        embed_cat_feats = []
        for cat_col in self.args.cat_cols:
            value = data[cat_col]
            embed_cat_feat = getattr(self, f'embedding_{cat_col}')(value.int()) # self.embedding_xxx(xxx.int())
            embed_cat_feats.append(embed_cat_feat)
        embed = torch.cat([embed_interaction,*embed_cat_feats],dim=2) # dim=2는 3차원을 합친다는 의미
        # [승준] encoder embed 추가
        enc_embed = torch.cat([*embed_cat_feats],dim=2)

        # continious concat
        con_feats = []
        for con_col in self.args.con_cols: 
            value = data[con_col]
            con_feats.append(value.unsqueeze(2))
        embed = torch.cat([embed,*con_feats], dim=2).float()
        ################# [건우] ###############
        # [승준] encoder embed concat
        enc_embed = torch.cat([enc_embed,*con_feats], dim=2).float()

        X = self.comb_proj(embed) # concat후 feature_size=Hidden dimension으로 선형변환
        # [승준] encoder embed proj 추가
        enc_X = self.enc_comb_proj(enc_embed)
        # [찬우] LastQuery 모델의 positional_encoding을 위한 seq_len 추가
        return enc_X, X, batch_size, seq_len # embedding을 concat하고 선형 변환한 값


class LSTM(ModelBase):
    def __init__(self, args): # [건우] : args로 몰았기 때문에 args만 씀
        super().__init__(args) # [건우] : args로 몰았기 때문에 args만 씀
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True # input_size=hidden_size
        ) 

    def forward(self, data):
        # X는 embedding들을 concat한 값
        # super().forward은 부모객체의 forward메소드를 말함
        _, X, batch_size, _ = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음
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
        _, X, batch_size, _ = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음

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
        _, X, batch_size, _ = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=data["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


##LastQuery 추가


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True)


    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, data):
        _, X, batch_size, seq_len = super().forward(data)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################

        q = self.query(X).permute(1, 0, 2)
        
        
        q = self.query(X)[:, -1:, :].permute(1, 0, 2)
        
        
        
        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = X + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = X + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))
        
        # input embedding
        pe = torch.zeros(max_len, d_model) ## max_len X hidden_dim
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #0부터 sequence 길이만큼 position 값 생성, 1 X max_len
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(ModelBase):

    def __init__(self, args):
        super(Saint, self).__init__(args)
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        self.device = args.device
       
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.max_seq_len)
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.drop_out,
            activation='relu')

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, data):
        
        seq_emb_enc, seq_emb_dec, batch_size, seq_len = super().forward(data)
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device).to(torch.float32)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device).to(torch.float32)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device).to(torch.float32)

        seq_emb_enc = seq_emb_enc.permute(1, 0, 2)
        seq_emb_dec = seq_emb_dec.permute(1, 0, 2)

        # Positional encoding custum
        seq_emb_enc = self.pos_encoder(seq_emb_enc)
        seq_emb_dec = self.pos_decoder(seq_emb_dec)

        out = self.transformer(seq_emb_enc, seq_emb_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)
  
        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        
        return out

class FixupEncoder(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        
        # Defining some parameters
        self.Tfixup = self.args.Tfixup
        self.layer_norm = self.args.Tfix_layer_norm
        self.n_layers = self.args.Tfix_n_layers
        
        # Encoder
        self.encoders = nn.ModuleList([EncoderLayer(args) for _ in range(self.n_layers)])
        
        # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param          
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def mask_2d_to_3d(self, mask, batch_size, seq_len):
        # padding 부분에 1을 주기 위해 0과 1을 뒤집는다
        mask = torch.ones_like(mask) - mask
        
        mask = mask.repeat(1, seq_len)
        mask = mask.view(batch_size, -1, seq_len)
        mask = mask.repeat(1, self.args.n_heads, 1)
        mask = mask.view(batch_size*self.args.n_heads, -1, seq_len)

        return mask.float().masked_fill(mask==1, float('-inf'))

    def forward(self, data):
        _, X, batch_size, seq_len = super().forward(data)
        mask = data['mask']

        ### Encoder
        mask = self.mask_2d_to_3d(mask, batch_size, seq_len).to(self.device)
        for encoder in self.encoders:
            X = encoder(X, mask)

        ###################### DNN #####################
        out = X.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.Tfix_n_layers
        self.layer_norm = self.args.Tfix_layer_norm

        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)

        self.ffn1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.ffn2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)   

        if self.layer_norm:
            self.ln1 = nn.LayerNorm(self.hidden_dim)
            self.ln2 = nn.LayerNorm(self.hidden_dim)


    def forward(self, embed, mask):
        q = self.query(embed).permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        out, _ = self.attn(q, k, v, attn_mask=mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        
        if self.layer_norm:
            out = self.ln1(out)

        ## feed forward network
        out = self.ffn1(out)
        out = F.relu(out)
        out = self.ffn2(out)

        ## residual + layer norm
        out = embed + out

        if self.layer_norm:
            out = self.ln2(out)

        return out
