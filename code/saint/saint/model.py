import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from sklearn.preprocessing import StandardScaler

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

class Saint(nn.Module):

    def __init__(self, args):
        super(Saint, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.dropout
        #self.dropout = 0.

        ### Embedding
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # encoder combination projection
        self.enc_cate_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim//2)
        self.enc_cont_comb_proj = nn.Linear(2, self.hidden_dim//2) ## 임시로 현재 3개로 지정, 추후 코드 변경 필요, cont column의 개수임
        
        
        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)

        # decoder combination projection
        self.dec_cate_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim//2)
        self.dec_cont_comb_proj = nn.Linear(2, self.hidden_dim//2)## 임시로 현재 3개로 지정, 추후 코드 변경 필요, cont column의 개수임
       
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        
        test, question, tag, _, mask, elapsed, KnowledgeTag_percent, cumulative, paper_number, interaction, _ = input


        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # embedding
        # ENCODER

        # cate embedding & concat
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        
        
        cate_embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag,], 2)
        # cate projection
        cate_embed_enc = self.enc_cate_comb_proj(cate_embed_enc)

        # cont embedding & concat
        cont_embed_enc = torch.cat([#elapsed,
                               KnowledgeTag_percent,
                               cumulative,], 1)
        
        cont_embed_enc = cont_embed_enc.view(batch_size, seq_len, -1) # (batch_size , seq_len, cont_col.size())
        
        # cont projection
        cont_embed_enc = self.enc_cont_comb_proj(cont_embed_enc)

        # cate, cont combine
        seq_emb_enc = torch.cat([cate_embed_enc,cont_embed_enc],2)

        # DECODER
        # cate embedding & concat
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)
       
        cate_embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction], 2)
        # cate projection
        cate_embed_dec = self.dec_cate_comb_proj(cate_embed_dec)
        
        # cont embedding & concat
        cont_embed_dec = torch.cat([#elapsed,
                               KnowledgeTag_percent,
                               cumulative,], 1)
        
        cont_embed_dec = cont_embed_dec.view(batch_size, seq_len, -1) # (batch_size , seq_len, cont_col.size())

        # cont projection
        cont_embed_dec = self.dec_cont_comb_proj(cont_embed_dec)
        
        # cate, cont combine
        seq_emb_dec = torch.cat([cate_embed_dec, cont_embed_dec],2)

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
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)
        
        return preds