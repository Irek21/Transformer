import torch
import torch.nn as nn
import math

PAD_DE = 1
BOS_DE = 16384
EOS_DE = 16385

PAD_EN = 0
BOS_EN = 16384
EOS_EN = 16385

def gen_mask(seq_size):
    mask = (torch.triu(torch.ones(seq_size, seq_size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = gen_mask(tgt_seq_len).to('cuda')
    src_mask = torch.zeros((src_seq_len, src_seq_len), device='cuda').type(torch.bool)

    src_padding_mask = (src == PAD_DE).transpose(0, 1).to('cuda')
    tgt_padding_mask = (tgt == PAD_EN).transpose(0, 1).to('cuda')
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

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
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, 
                 nlayers_enc, nlayers_dec, d_fforward, 
                 vsize1, vsize2,
                 dropout):
        super(Transformer, self).__init__()
        
        self.vsize1 = vsize1
        # self.embed1 = nn.Embedding(num_embeddings=vsize1, embedding_dim=d_model)

        self.vsize2 = vsize2
        # self.embed2 = nn.Embedding(num_embeddings=vsize2, embedding_dim=d_model)
        self.embed1 = TokenEmbedding(vsize1, d_model)
        self.embed2 = TokenEmbedding(vsize2, d_model)

        self.pos_enc = PositionalEncoding(emb_size=d_model, dropout=dropout)
        
        self.d_model = d_model
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout,
                                                    dim_feedforward=d_fforward)
        self.enc = nn.TransformerEncoder(encoder_layer=self.enc_layer,
                                         num_layers=nlayers_enc)
        
        self.encdec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                       dropout=dropout, 
                                                       dim_feedforward=d_fforward)
        self.dec = nn.TransformerDecoder(decoder_layer=self.encdec_layer, 
                                         num_layers=nlayers_dec)
        
        self.fc = nn.Linear(d_model, vsize2)

    def forward(self, src, tgt):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

        src = self.embed1(src) #  * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        tgt = self.embed2(tgt) #  * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        
        out = self.enc(src,
                       mask=src_mask,
                       src_key_padding_mask=src_padding_mask)
        out = self.dec(tgt, out,
                       tgt_mask=tgt_mask,
                       memory_key_padding_mask=src_padding_mask,
                       tgt_key_padding_mask=tgt_padding_mask)
        
        out = self.fc(out)
        return out

    def encode(self, src, src_mask):
        return self.enc(self.pos_enc(self.embed1(src)), #  * math.sqrt(self.d_model), 
                        src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.dec(self.pos_enc(self.embed2(tgt)), # * math.sqrt(self.d_model), 
                        memory, tgt_mask)