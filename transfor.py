import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig  # 可根据需要替换为其他Transformer模型
from torch.nn.utils.rnn import pad_sequence
import math
vocab = [str(i) for i in range(1000)]
vocab.extend(['[UNK]', '[PAD]', '[ST]', '[ED]','SEP','**','*','+','-','/','exp','E','PI','cos','sin','tan','sqrt','log','t','10^','x0','x1'])
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=word_to_idx['[PAD]'])
    tgt_batch = pad_sequence(src_batch, batch_first=True, padding_value=word_to_idx['[PAD]'])
    return src_batch, tgt_batch

def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).to(torch.bool)
    return mask

def create_padding_mask(seq, pad_idx):
    return (seq == pad_idx).to(torch.bool)

def lr_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return (warmup_steps / float(current_step)) ** 0.5

class TextDataset(Dataset):
    def __init__(self, questions_idx, answers_idx, max_len):
        self.questions = [q + [word_to_idx['[PAD]']] * (max_len - len(q)) for q in questions_idx]
        self.answers = [a + [word_to_idx['[PAD]']] * (max_len - len(a)) for a in answers_idx]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        src = torch.tensor(self.questions[idx], dtype=torch.long)  # 问题输入
        tgt = torch.tensor(self.answers[idx], dtype=torch.long)    # 答案输出
        return src, tgt

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embedding_dim), requires_grad=False)
        self.initialize_positional_encoding(embedding_dim, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=False
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def initialize_positional_encoding(self, embedding_dim, max_seq_len):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, src, tgt):
        # 生成自回归屏蔽矩阵
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(src.device)
    
        # 创建填充屏蔽
        src_padding_mask = create_padding_mask(src, word_to_idx['[PAD]']).to(src.device)
        tgt_padding_mask = create_padding_mask(tgt, word_to_idx['[PAD]']).to(tgt.device)

        # 添加嵌入和位置编码
        src = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :]

        # 转编以适应 Transformer 模块
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # 前向传播时，传递握取
        output = self.transformer(
            src, tgt, tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask
        )
    
        # 取出模型输出并进行最后的线性映射
        output = output.transpose(0, 1)
        output = self.fc_out(output)
        return output

