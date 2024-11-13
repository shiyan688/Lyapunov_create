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
        elif token == 'cos' or token == 'sin':
            self.position += 1
            argument = self.parse_expression()
            return f"{token}({argument})"
        else:
            # Assuming token is a number or variable
            self.position += 1
            return token

def sequence_to_equation(sequence):
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
tokens = ['*', 'cos', '*', '2.1', 'x0', '+', 'x1', '2', 'SEP', 'sin', '+', '*', '3', 'x1', '2']
equation = sequence_to_equation(tokens)
print(equation)
