import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig  # 根据需要更换为其他Transformer模型
from torch.nn.utils.rnn import pad_sequence

# 定义TransformerModel类 (从原始代码中拷贝)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 修改位置编码器的初始化，使用大小和保存权重一致
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
        
        output = self.transformer(src, tgt)
        
        # 将输出维度调整回来
        output = output.transpose(0, 1)
        output = self.fc_out(output)
        return output

# 加载词汇表和模型参数
vocab = [...]  # 使用与训练时相同的词汇表
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size = len(vocab)
embedding_dim = 128
num_heads = 8
num_layers = 2
hidden_dim = 256
max_seq_len = 2155  # 使用与训练时相同的最大序列长度

# 初始化模型并加载保存的权重
model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len)
# 使用strict=False来加载不完全匹配的参数
model.load_state_dict(torch.load('transformer_polish_model.pth'), strict=False)
model = model.cuda() if torch.cuda.is_available() else model
model.eval()  # 切换到评估模式

# 输入句子并进行推理
def preprocess_input(input_text, word_to_idx):
    input_tokens = input_text.split(', ')
    input_idx = [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in input_tokens]
    input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).cuda() if torch.cuda.is_available() else torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
    return input_tensor

input_text = "[+, *, **, **, x2, x0, 50, 10^, 2, ...]"  # 这是一个示例输入
input_tensor = preprocess_input(input_text, word_to_idx)

# 使用模型进行推理
with torch.no_grad():
    # 假设目标输入与问题一致，用于简单测试
    output = model(input_tensor, input_tensor)
    output_idx = torch.argmax(output, dim=-1)

# 将输出索引转换为词汇
output_tokens = [idx_to_word[idx.item()] for idx in output_idx[0]]
output_text = ', '.join(output_tokens)
print(f"模型生成的输出：{output_text}")
