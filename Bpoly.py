import numpy as np
import sympy as sp
import random  


# 定义可用的二元和一元运算符
binary_operators = ['+', '-', '*', '/', '**']  # 二元运算符
unary_operators = ['exp', 'log', 'sqrt', 'sin', 'cos', 'tan']  # 一元运算符
def generate_random_leaf(variables):
    """
    随机生成叶子节点，可以是变量或整数。
    """
    if random.random() < 0.6:
        return random.choice(variables)  # 变量
    else:
        return sp.Integer(random.randint(1, 10))  # 随机整数

def generate_random_function(variables, depth=2):
    """
    递归生成随机函数，并确保生成的函数不包含复数或无穷大。
    
    参数：
    - variables: 符号变量列表
    - depth: 当前递归深度，用于控制树的高度
    """
    while True:
        # 生成随机函数
        if depth == 0:
            expr = generate_random_leaf(variables)
        else:
            if random.random() < 0.5:
                # 生成二元运算符
                operator = random.choice(binary_operators)
                left = generate_random_function(variables, depth - 1)
                right = generate_random_function(variables, depth - 1)
                expr = sp.sympify(f"({left}) {operator} ({right})")
            else:
                # 生成一元运算符
                operator = random.choice(unary_operators)
                operand = generate_random_function(variables, depth - 1)
                expr = sp.sympify(f"{operator}({operand})")

        # 检查是否包含复数或无穷大
        if expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.I) or expr.has(sp.zoo) or expr.has(-sp.nan):
            print("Detected invalid expression (complex or infinity), regenerating...")
            continue  # 如果包含无穷大或复数，则重新生成表达式
        else:
            return expr  # 如果没有无穷大或复数，返回生成的表达式

# 生成多项式函数
def generate_random_polynomial(variables, num_terms=3, coeff_range=(-10, 10), power_range=(1, 3)):
    """
    生成一个随机多项式，由多个随机单项式组成。
    
    参数：
    - variables: 符号变量列表（多项式的变量）
    - num_terms: 多项式中包含的单项式的数量
    - coeff_range: 系数的范围，默认在 -10 到 10 之间
    - power_range: 幂次的范围，默认在 1 到 3 之间
    
    返回：
    - 随机生成的多项式
    """
    polynomial = 0
    for _ in range(num_terms):
        coeff = random.randint(*coeff_range)
        term = coeff
        for var in variables:
            power = random.randint(*power_range)
            term *= var**power
        polynomial += term
    
    return polynomial
def generate_random_positive_definite_matrix(n):
    """
    生成一个随机正定矩阵，可以设置一定概率为对角矩阵。
    """
    # 生成随机矩阵
    A = np.random.rand(n, n)
    A = np.dot(A, A.transpose())  # 确保正定矩阵
    
    # 随机决定是否为对角矩阵
    if random.random() < 0.5:
        A = np.diag(np.diag(A))
    
    return A
# Step 1: 生成李雅普诺夫函数
def generate_lyapunov_function(n_vars, depth=3):
    """
    随机生成一个满足要求的李雅普诺夫函数 V = V_cross + V_proper。
    
    V_cross 是若干项 pi(x)^2 的和，满足 V_cross(0) = 0。
    V_proper 是一个给定的正定函数。
    
    参数：
    - n_vars: 变量数量
    - depth: 随机生成函数的深度
    """
    # 生成符号变量
    x = sp.symbols(f'x:{n_vars}')
    
    # 生成 V_cross，V_cross(x) = sum(pi(x)^2) 形式，其中 pi(x) 是随机函数
    V_cross = 0
    for i in range(n_vars):
        pi = generate_random_polynomial(list(x), num_terms=1, coeff_range=(-10, 10), power_range=(1, 2))  # 生成随机函数 pi(x)
        V_cross += pi**2  # V_cross 是若干项 pi(x)^2 的和
    A = generate_random_positive_definite_matrix(n_vars)
    # 生成 V_proper，正定函数
    V_proper = 0
    for i in range(n_vars):
        for j in range(n_vars):
            alpha_ij = A[i, j]  # 从正定矩阵中获取 alpha_ij
            beta_i = random.randint(1, 3)
            beta_j = random.randint(1, 3)
            V_proper += alpha_ij * (x[i]**beta_i) * (x[j]**beta_j)
    print(V_cross + V_proper)
    # 返回 V = V_cross + V_proper
    return V_cross + V_proper

# Step 2: 计算梯度 ∇V 以及随机符号化向量
def compute_gradient(V, x):
    """
    计算李雅普诺夫函数 V(x) 的梯度。
    """
    grad_V = sp.Matrix([sp.diff(V, xi) for xi in x])
    return grad_V
def generate_random_vectors(n_vars, num_vectors):
    """
    生成一组随机的符号向量。
    """
    vectors = []
    for _ in range(num_vectors):
        vec = sp.Matrix([sp.Rational(sp.randprime(1, 105), 1) * x[i] for i in range(n_vars)])  # 使用符号向量
        vectors.append(vec)
    return vectors
# Step 3: 使用格拉姆-施密特正交化方法生成与梯度超平面正交的基向量
def gram_schmidt_orthogonalization(grad_V, vectors):
    """
    使用格拉姆-施密特正交化方法，生成与梯度超平面正交的基向量。
    """
    
    def project(u, v):
        """将向量 v 投影到向量 u 上"""
        
        return (v.T * u / (u.T * u)[0])[0] * u  # 符号化计算

    orthogonal_vectors = []
    for v in vectors:
        # 对每个向量执行投影消除，确保它们相互正交
        for u in orthogonal_vectors:
            v = v - project(u, v)
        # 使当前向量与梯度正交
        v = v - project(grad_V, v)
        
        orthogonal_vectors.append(v)
    
    return orthogonal_vectors
# Step 4: 采样

# Step 4: 生成动力系统 f(x)
def generate_dynamical_system(V, grad_V, orth_vectors):
    """
    生成动力系统方程，使得李雅普诺夫函数 V 是该系统的李雅普诺夫函数。
    """
    n = len(grad_V)
    # Step 1: 随机选择 1 <= p <= n，从正交向量中采样 p 个向量
    p = random.randint(1, n - 1)  # 随机选择 p 的值
    sampled_orth_vectors = random.sample(orth_vectors, p)  # 从 orth_vectors 中采样 p 个向量
    
    # Step 2: 随机选择 1 <= k1 <= n，生成 k1 个实值函数 h_i(x)，并将 h_i = 0 对于 k1+1 <= i <= n
    k1 = random.randint(1, n)  # 随机选择 k1 的值
    h_functions = [generate_random_function(list(sp.symbols(f'x:{n}'))) for _ in range(k1)]  # 生成 k1 个实值函数
    h_functions += [sp.Integer(0) for _ in range(k1, n)]  # 对于 k1+1 <= i <= n，设置 h_i = 0
    
    # 动力系统方程的第一部分：-h(x)^2 * ∇V(x)
    f_grad = sp.Matrix([-(h_functions[i]**2) * grad_V[i] for i in range(n)])  # 梯度项
    
    # Step 3: 生成 g_i(x) * e^i(x) 的部分
    g_functions = [generate_random_function(list(sp.symbols(f'x:{n}'))) for _ in range(p)]
    f_orth = sp.zeros(n, 1)
    for i in range(p):
        f_orth += g_functions[i] * sampled_orth_vectors[i]  
    # 返回动力系统的最终方程
    return f_grad + f_orth

# 主函数：反向生成动力系统和李雅普诺夫函数
def backward_generation(n_vars):
    x = sp.symbols(f'x:{n_vars}')
    V = generate_lyapunov_function(n_vars)
    grad_V = compute_gradient(V, x)
    random_vectors = generate_random_vectors(n_vars, n_vars-1)
    orth_vectors = gram_schmidt_orthogonalization(grad_V, random_vectors)
    f_system = generate_dynamical_system(V, grad_V, orth_vectors)
    return V, f_system
def save_vectors_to_file(vectors, filename):
    """
    将生成的向量写入到txt文件中。
    
    参数：
    - vectors: 要保存的向量列表
    - filename: 保存的文件名
    """
    with open(filename, 'w') as f:
        for vec in vectors:
            f.write(str(vec) + '\n')
    print(f"向量已保存至 {filename}")
def save_functions_to_file(functions, filename):
    
    with open(filename, 'w') as f:
        
        f.write(str(functions) + '\n')
    print(f"functions已保存至 {filename}")

# 运行主函数
n_vars = 3  # 变量数量
x = sp.symbols(f'x:{n_vars}')

V = x[0]**2 + x[1]**2 + x[2]**2  # 指定的 V
grad_V = compute_gradient(V, x)
random_vectors = generate_random_vectors(n_vars, n_vars - 1)
orth_vectors = gram_schmidt_orthogonalization(grad_V,  random_vectors)

f_system = generate_dynamical_system(V, grad_V, orth_vectors)
f_system = sp.sympify(f_system)
save_functions_to_file(V, "lyapunov_function_for.txt")
save_vectors_to_file(f_system, "dynamical_system_for.txt")
