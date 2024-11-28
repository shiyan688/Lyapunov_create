class ExpressionParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def parse(self):
        return self.parse_expression()

    def parse_expression(self):
        if self.position >= len(self.tokens):
            return None

        token = self.tokens[self.position]
        if token == '*':
            self.position += 1
            left = self.parse_expression()
            right = self.parse_expression()
            return f"({left} * {right})"
        elif token == '+':
            self.position += 1
            left = self.parse_expression()
            right = self.parse_expression()
            return f"({left} + {right})"
        elif token == '**':
            self.position += 1
            left = self.parse_expression()
            right = self.parse_expression()
            return f"({left} ^ {right})"
        elif token == 'cos' or token == 'sin' or token =='tan' or token =='sqrt' or token =='log' or token=='-':
            self.position += 1
            argument = self.parse_expression()
            return f"{token}({argument})"
        else:
            # Assuming token is a number or variable
            self.position += 1
            return token

def preprocess_tokens(tokens):
    processed_tokens = []
    i = 0
    read_tokens = tokens
    while i < len(tokens):
        if tokens[i] == 't':
            # 如果当前标记是 't'，将其与前一个和后一个标记连接起来
            if i > 0 and i < len(tokens) - 1:
                if tokens[i+1].isdigit():
                # 如果是数字标记，根据位数进行处理
                    if len(tokens[i + 1]) == 1:
                        tokens[i + 1] = '00' + tokens[i + 1]  # 对一位数添加两个前导零
                    elif len(tokens[i + 1]) == 2:
                        tokens[i + 1] = '0' + tokens[i + 1]  # 对两位数添加一个前导零
                new_token = processed_tokens[-1] + tokens[i + 1]
                processed_tokens.pop()  # 移除前一个标记
                processed_tokens.append(new_token)
                
                i += 1  # 跳过后一个标记
            else:
                # 如果 't' 没有前后标记，保持原样
                processed_tokens.append(tokens[i])
        elif tokens[i] == '10^':
            # 如果当前标记是 '10^'，检查下一个标记是否是 '-'
            if i < len(tokens) - 2 and tokens[i + 1] == '-':
                # 将前一个标记、'10^'、'-' 和其后的标记连接起来
                if i > 0:
                    new_token = processed_tokens[-1]+'*' + tokens[i] + tokens[i + 1] + tokens[i + 2]
                    processed_tokens.pop()  # 移除前一个标记
                    processed_tokens.append(new_token)
                    i += 2  # 跳过 '-' 和其后的标记
                else:
                    # 如果 '10^' 没有前一个标记，保持原样
                    processed_tokens.append(tokens[i])
            elif i < len(tokens) - 1:
                # 否则将前一个标记、'10^' 和其后的标记连接起来
                if i > 0:
                    new_token = processed_tokens[-1]+'*' + tokens[i] + tokens[i + 1]
                    processed_tokens.pop()  # 移除前一个标记
                    processed_tokens.append(new_token)
                    i += 1  # 跳过后一个标记
                else:
                    # 如果 '10^' 没有前一个标记，保持原样
                    processed_tokens.append(tokens[i])
        
        else:
            # 对于其他标记，直接添加到处理后的列表中
            processed_tokens.append(tokens[i])
        
        i += 1

    return processed_tokens

def sequence_to_equation(sequence):
    # 预处理标记列表
    sequence = preprocess_tokens(sequence)
    
    equations = []
    while 'SEP' in sequence:
        sep_index = sequence.index('SEP')
        
        sub_sequence = sequence[:sep_index]
        
        parser = ExpressionParser(sub_sequence)
        equations.append(parser.parse())
        sequence = sequence[sep_index + 1:]
    # Parse the remaining sequence after the last SEP
    if sequence:
        parser = ExpressionParser(sequence)
        equations.append(parser.parse())
    return "\n".join(equations)


# Example usage
def main():
    with open('dynamical_system_polish4.txt', 'r') as f:
        question_lines = f.readlines()
        
# 读取答案文件
    with open('lyapunov_function_polish_poly4.txt', 'r') as f:
        answer_lines = f.readlines()
    for line in question_lines[:1]:
        line = line.strip().strip('[]').strip().strip(',')
        line_tokens = line.split(', ')
        line_tokens = line_tokens[:-1] 
        
        equation = sequence_to_equation(line_tokens)
        print(equation)  
    for line in answer_lines[:1]:
        line = line.strip().strip('[]').strip().strip(',')
        line_tokens = line.split(', ')
        equation = sequence_to_equation(line_tokens)
        print(equation)     
    return 0
if __name__ == "__main__":
    main()

