# 文件: transformer_polish_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig  # 可根据需要替换为其他Transformer模型
from torch.nn.utils.rnn import pad_sequence
import math
import transfor as tr

if __name__ == "__main__":
    # 读取问题文件
    with open('dynamical_system_polish_poly.txt', 'r') as f:
        question_lines = f.readlines()

    # 读取答案文件
    with open('lyapunov_function_polish_poly.txt', 'r') as f:
        answer_lines = f.readlines()

    # 遍历问题和答案文件的每一行
    tokens = []
    for line in question_lines + answer_lines:
        line = line.strip().strip('[]').strip().strip(',')
        
        line_tokens = line.split(', ')
        tokens.extend([token.strip() for token in line_tokens])

    # 创建词表
    vocab = [str(i) for i in range(1000)]
    vocab.extend(['[UNK]', '[PAD]', '[ST]', '[ED]','SEP','**','*','+','-','/','exp','E','PI','cos','sin','tan','sqrt','log','t','10^','x0','x1','10^'])
    print(vocab)

    # 创建词汇到索引的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # 将问题和答案数据转换为索引
    questions_idx = []
    answers_idx = []

    for line in question_lines:
        line = line.strip().strip('[]').strip().strip(',')
        line_tokens = line.split(', ')
        
        idxs = [word_to_idx['[ST]']] + [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens] + [word_to_idx['[ED]']]
        questions_idx.append(idxs)

    for line in answer_lines:
        line = line.strip().strip('[]').strip().strip(',')
        line_tokens = line.split(', ')
        idxs = [word_to_idx['[ST]']] + [word_to_idx.get(token.strip(), word_to_idx['[UNK]']) for token in line_tokens] + [word_to_idx['[ED]']]
        answers_idx.append(idxs)

    # 检查问题和答案数量是否一致
    if len(questions_idx) != len(answers_idx):
        print(f"Mismatch: questions count = {len(questions_idx)}, answers count = {len(answers_idx)}")
        questions_idx = questions_idx[:1000]
        answers_idx = answers_idx[:1000]
        print(f"Using first 1000 pairs for training.")
    else:
        questions_idx = questions_idx[:2000]
        answers_idx = answers_idx[:2000]
        print(f"Questions and answers are matched: {len(questions_idx)} pairs")
        

    max_seq_len = max(max(len(seq) for seq in questions_idx), max(len(seq) for seq in answers_idx))
    print(f"max_seq_len: {max_seq_len}")

    # 初始化数据集和数据加载器
    dataset = tr.TextDataset(questions_idx, answers_idx, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    vocab_size = len(vocab)
    embedding_dim = 256
    num_heads = 8
    num_layers = 2
    hidden_dim = 256

    # 初始化模型、损失函数和优化器
    model = tr.TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['[PAD]'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义学习率调度器，包含预热阶段（10,000 步）和逆平方根衰减
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=tr.lr_lambda)
    print("start")

    # 训练模型
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (src, tgt) in enumerate(dataloader):
            src, tgt = src.cuda(), tgt.cuda()
            optimizer.zero_grad()
            
            output = model(src, tgt[:, :-1])  
            output = output.view(-1, vocab_size)
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'max_seq_len': max_seq_len
    }, 'transformer_polish_model_hope1.pth')


# 定义其他模块和函数
