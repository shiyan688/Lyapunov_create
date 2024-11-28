import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import ast
from torch.nn.utils.rnn import pad_sequence
from transfor import TransformerModel,TextDataset,collate_fn

def generate_answer(model, test_src):
    # 初始化目标序列为空
    generated_tokens = []
    tgt = torch.tensor([word_to_idx['[ST]']]).unsqueeze(0).to(test_src.device)

    # 开始生成答案
    with torch.no_grad():
        for _ in range(saved_max_seq_len):
            # 模型前向推理
            output = model(test_src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1)  # 获取最大概率的下一个词
            generated_tokens.append(next_token.item())
            
            # 如果生成了结束标记，停止生成
            if next_token.item() == word_to_idx['[ED]']:
                break
            
            # 更新目标序列
            tgt = torch.cat((tgt, next_token.unsqueeze(0)), dim=1)

    # 将生成的索引转换为单词
    generated_answer = [idx_to_word[idx] for idx in generated_tokens]
    return " ".join(generated_answer)


checkpoint = torch.load('transformer_polish_model_hope1.pth')
vocab = checkpoint['vocab']
saved_max_seq_len = checkpoint['max_seq_len']

# 确保 [UNK] 和 [PAD] 在词汇中
if '[UNK]' not in vocab:
    vocab.add('[UNK]')
if '[PAD]' not in vocab:
    vocab.add('[PAD]')

# 创建词汇到索引和索引到词汇的映射
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 读取测试集数据
with open('dynamical_system_polish_poly.txt', 'r') as f:
    test_question_lines = f.readlines()[:1000]

with open('lyapunov_function_polish_poly.txt', 'r') as f:
    test_answer_lines = f.readlines()[:1000]

# 将测试数据转换为索引
test_questions_idx = []
test_answers_idx = []

for line in test_question_lines:
    line = line.strip().strip('[]').strip().strip(',')
    line_tokens = line.split(', ')
    idxs = [word_to_idx['[ST]']] + [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens] + [word_to_idx['[ED]']]
    test_questions_idx.append(idxs)

for line in test_answer_lines:
    line = line.strip().strip('[]').strip().strip(',')
    line_tokens = line.split(', ')
    idxs = [word_to_idx['[ST]']] + [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens] + [word_to_idx['[ED]']]
    test_answers_idx.append(idxs)

# 计算最大序列长度


# 从checkpoint中加载模型并获取max_seq_len

# 初始化模型
vocab_size = len(vocab)
embedding_dim = 256
num_heads = 8
num_layers = 2
hidden_dim = 256

model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, saved_max_seq_len)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

# 定义填充函数

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
test_dataset = TextDataset(test_questions_idx, test_answers_idx,saved_max_seq_len)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 损失函数
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['[PAD]'])

# 在测试集上评估模型
total_test_loss = 0
with torch.no_grad():
    for src, tgt in test_dataloader:
        src, tgt = src.cuda(), tgt.cuda()
        output = model(src, tgt[:, :-1])
        output = output.view(-1, vocab_size)
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        total_test_loss += loss.item()

print(f"Test Loss: {total_test_loss / len(test_dataloader)}")

# 从测试集生成答案


# 从测试集中随机选择一个样本进行推理

random_idx = random.randint(0, len(test_questions_idx) - 1)
question_tensor = torch.tensor(test_questions_idx[random_idx], dtype=torch.long)

# 使用 pad_sequence 并将结果移到 GPU 上
test_src = pad_sequence([question_tensor], batch_first=True, padding_value=word_to_idx['[PAD]']).cuda()
original_question = " ".join([idx_to_word[idx] for idx in test_questions_idx[random_idx]])
generated_answer = generate_answer(model, test_src)
generated_answer = "[ST] " + generated_answer
print("Original Question:", original_question)
print("Generated Answer:", generated_answer)
