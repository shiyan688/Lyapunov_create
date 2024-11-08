import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig  # 可根据需要更换为其他Transformer模型
from torch.nn.utils.rnn import pad_sequence
import math

# 读取问题文件
with open('dynamical_system_polish.txt', 'r') as f:
    question_lines = f.readlines()

# 读取答案文件
with open('lyapunov_function_polish.txt', 'r') as f:
    answer_lines = f.readlines()

# 初始化一个空列表来存储所有的词汇
tokens = []
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=word_to_idx['[PAD]'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=word_to_idx['[PAD]'])
    return src_batch, tgt_batch
# 遍历问题和答案文件的每一行
for line in question_lines + answer_lines:
    # 去掉行首尾空格并去掉列表符号 []
    line = line.strip('[]').strip()
    # 分割成单独的词汇
    line_tokens = line.split(', ')
    # 去除词汇的空格
    tokens.extend([token.strip() for token in line_tokens])

# 创建词表
vocab = set(tokens)
vocab.add('[PAD]')
vocab.add('[UNK]')
print(vocab)

# 创建词汇到索引的映射
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 将问题和答案数据转换为索引
questions_idx = []
answers_idx = []

for line in question_lines:
    line = line.strip('[]').strip()
    line_tokens = line.split(', ')
    idxs = [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens]
    questions_idx.append(idxs)

for line in answer_lines:
    line = line.strip('[]').strip()
    line_tokens = line.split(', ')
    idxs = [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens]
    answers_idx.append(idxs)

# 确保问题和答案数量匹配
# 检查问题和答案数量是否一致
if len(questions_idx) != len(answers_idx):
    print(f"Mismatch: questions count = {len(questions_idx)}, answers count = {len(answers_idx)}")
    # 取前1000个进行训练
    questions_idx = questions_idx[:1000]
    answers_idx = answers_idx[:1000]
    print(f"Using first 1000 pairs for training.")
else:
    questions_idx = questions_idx[:100]
    answers_idx = answers_idx[:100]
    print(f"Questions and answers are matched: {len(questions_idx)} pairs")

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

max_seq_len = max(max(len(seq) for seq in questions_idx), max(len(seq) for seq in answers_idx))
print(f": max_seq_len{max_seq_len} ")

# 初始化数据集和数据加载器
# 初始化数据集和数据加载器
dataset = TextDataset(questions_idx, answers_idx, max_seq_len)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=False  # 关闭 batch_first
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt):
        # 添加位置编码
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        
        # 调整维度顺序以符合 Transformer 的输入要求 (seq_len, batch_size, embedding_dim)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        #print(f"src shape: {src.shape}, tgt shape: {tgt.shape}")

        output = self.transformer(src, tgt)
        
        # 将输出维度调整回来
        output = output.transpose(0, 1)
        output = self.fc_out(output)
        return output

vocab_size = len(vocab)  # 词汇表大小
embedding_dim = 128  # 嵌入维度
num_heads = 8  # 多头注意力头数
num_layers = 2  # Transformer层数
hidden_dim = 256  # 前馈神经网络的隐藏层维度

# 初始化模型、损失函数和优化器
model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len)
model = model.cuda()  # 将模型转到GPU上
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['[PAD]'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.cuda(), tgt.cuda()  # 将输入和标签转到GPU
        optimizer.zero_grad()
        
        # 使用 tgt[:, :-1] 作为输入，tgt[:, 1:] 作为标签，确保长度一致
        output = model(src, tgt[:, :-1])  
        output = output.view(-1, vocab_size)
        tgt = tgt[:, 1:].contiguous().view(-1)  # 目标序列偏移一个标记作为标签

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")
# 保存模型
torch.save(model.state_dict(), 'transformer_polish_model.pth')
