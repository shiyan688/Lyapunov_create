import numpy as np
import sympy as sp
import random  
from mpmath import mp

mp.dps = 50  # 设置为 50 位有效数字

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
        if expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.I) or expr.has(sp.zoo) or expr.has(-sp.nan) or expr.has(sp.S.Zero):
            print("Detected invalid expression (complex or infinity), regenerating...")
            continue  # 如果包含无穷大或复数，则重新生成表达式
        else:
            return expr  # 如果没有无穷大或复数，返回生成的表达式
# 生成多项式函数
def generate_random_polynomial(variables, num_terms=2, coeff_range=(-5, 5), power_range=(1, 3)):
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
        pi = generate_random_polynomial(list(x))  # 生成随机函数 pi(x)
        V_cross += pi**2  # V_cross 是若干项 pi(x)^2 的和
    A = generate_random_positive_definite_matrix(n_vars)
    # 生成 V_proper，正定函数
    V_proper = 0
    beta = [random.randint(1, 3) for _ in range(n_vars)]
    for i in range(n_vars):
        for j in range(n_vars):
            alpha_ij = A[i, j]  # 从正定矩阵中获取 alpha_ij
            
            V_proper += alpha_ij * (x[i]**beta[i]) * (x[j]**beta[j])
    
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
        vec = sp.Matrix([sp.Rational(sp.randprime(1, 10), 1) * x[i] for i in range(n_vars)])  # 使用符号向量
        vectors.append(vec)
    return vectors
# Step 3: 使用格拉姆-施密特正交化方法生成与梯度超平面正交的基向量
def gram_schmidt_orthogonalization(grad_V, vectors):
   

    def project(u, v):
        """将向量 v 投影到向量 u 上"""
        

        # 计算 u 的长度平方
        u_norm_squared =u.dot(u)

        # 计算 v 在 u 上的投影系数
        projection_coefficient = u.dot(v) / u_norm_squared

        # 计算投影向量
        projection_vector = projection_coefficient * u # 返回投影向量

        return projection_vector

    orthogonal_vectors = []
    for v in vectors:
        # 对每个向量执行投影消除，确保它们相互正交
        
        # 使当前向量与梯度正交
        v = v - project(grad_V, v)
        
       
        orthogonal_vectors.append(v)
    
    return orthogonal_vectors
# Step 4: 采样
def check_f_orthogonality(grad_V, f_orth):
    """
    检查 f_orth 是否与 V 的梯度 grad_V 正交。
    返回 True 如果内积接近零，否则返回 False。
    """
    tolerance = 1e-6  # 设置容差
    dot_product = grad_V.dot(f_orth)  # 计算内积
    
    # 判断内积是否接近零
    if sp.simplify(dot_product).is_zero or abs(dot_product.evalf()) < tolerance:
        print("f_orth 与梯度 grad_V 正交内积为: {dot_product}")
        return True
    else:
        print(f"f_orth 与梯度 grad_V 不正交，内积为: {dot_product}")
        return False

# Step 4: 生成动力系统 f(x)
def generate_dynamical_system(V, grad_V, orth_vectors):
    """
    生成动力系统方程，使得李雅普诺夫函数 V 是该系统的李雅普诺夫函数。
    """
    n = len(orth_vectors)  # 设置 n 为 orth_vectors 的个数
    
    # Step 1: 随机选择 1 <= p <= n，从正交向量中采样 p 个向量
    
    if n > 1:
        p = random.randint(1, n)  # 随机选择 p 的值
    else:
    # 处理 n <= 1 的情况，例如设置 p 为一个默认值或引发自定义异常
      p = 1  # 或者其他适合的逻辑
    
    sampled_orth_vectors = random.sample(orth_vectors, p)  # 从 orth_vectors 中采样 p 个向量
    
    # Step 2: 随机选择 1 <= k1 <= n，生成 k1 个实值函数 h_i(x)，并将 h_i = 0 对于 k1+1 <= i <= n
    k1 = random.randint(1, n)  # 随机选择 k1 的值
    h_functions = [generate_random_function(list(sp.symbols(f'x:{n}')),depth=1) for _ in range(k1)]  # 生成 k1 个实值函数
    h_functions += [sp.Integer(0) for _ in range(k1, n)]  # 对于 k1+1 <= i <= n，设置 h_i = 0
    
    # 动力系统方程的第一部分：-h(x)^2 * ∇V(x)
    h_squared = sum(h_i**2 for h_i in h_functions)  # 将 h(x) 的各元素平方相加
    f_grad = -h_squared * grad_V  # 生成动力系统的梯度项
    # Step 3: 生成 g_i(x) * e^i(x) 的部分
    g_functions = [generate_random_function(list(sp.symbols(f'x:{n}'))) for _ in range(p)]
    f_orth = sp.zeros(n_vars, 1)
    for i in range(p):
        f_orth += g_functions[i]* sampled_orth_vectors[i]  
        
    # 返回动力系统的最终方程
    dynamical_system=f_grad +f_orth
    
    return dynamical_system

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
import sympy as sp

def to_polish_notation(expr):
    """
    将 SymPy 表达式转换为符合指定格式的波兰表示法。
    """
    result = []
    if expr.is_Number:
        # 处理数字（包括整数和实数）
        if expr == int(expr):  # 整数
            num = int(expr)
            if num >= 0:
                parts = []
                while num >= 1000:
                    parts.append(num % 1000)  # 获取最后 3 位
                    num //= 1000  # 去掉最后 3 位
                parts.append(num)  # 最后一组（剩余的部分，可能少于 3 位）
    
    # 返回结果时需要反转，因为我们是从右边开始划分的
                parts=parts[::-1]
                for part in parts:
                    result.append(part)

            else:
                parts = []
                result = ["-"]
                num=abs(num)
                while num >= 1000:
                    parts.append(num % 1000)  # 获取最后 3 位
                    num //= 1000  # 去掉最后 3 位
                parts.append(num)  # 最后一组（剩余的部分，可能少于 3 位）
    
    # 返回结果时需要反转，因为我们是从右边开始划分的
                parts=parts[::-1]
                for part in parts:
                    result.append(part)
                
        else:  # 实数
    # 将数字转换为科学计数法形式
            str_num = f"{float(expr):.6e}"
            coeff = float(str_num.split('e')[0])
            exp = int(str_num.split('e')[1])

    # 处理小数点位数
            coeff_str = str(coeff)
            if '.' in coeff_str:
        # 获取小数点后的位数
                decimal_places = len(coeff_str.split('.')[1])
            else:
                decimal_places = 0

            if decimal_places < 6:
        # 如果小数点后位数小于6, 按小数点位数乘以
                coeff = int(coeff * 10 ** (decimal_places))  # 将尾数乘以10^decimal_places，消除小数点
                exp -= decimal_places  # 更新指数，减去小数点后的位数
            else:
        # 如果小数点后位数大于或等于6，直接乘以10^6
                coeff = int(coeff * 10 ** 6)
                exp -= 6
            # 如果指数为负数，确保格式为 ["-", abs(exp)]
            
            if exp < 0:
                if coeff < 0:
                    if abs(coeff)<1000:
                        result = ["-",abs(coeff), "10^", "-", abs(exp)]
                    else:
                        a1 = abs(coeff)/1000
                        a1=int(a1)
                        a2=abs(coeff)%1000
                        if a1<1000:
                            result = ["-",a1,a2, "10^", "-", abs(exp)]
                        else:
                            a3 = a1/1000
                            a3=int(a3)
                            a1=a1%1000
                            result = ["-",a3,a1,a2, "10^", "-", abs(exp)]
                else:                   
                    if abs(coeff)<1000:
                        result = [coeff, "10^", abs(exp)]
                    else:
                        a1 = abs(coeff)/1000
                        a1=int(a1)
                        a2=abs(coeff)%1000
                        if a1<1000:
                            result = [a1,a2, "10^", abs(exp)]
                        else:
                            a3 = a1/1000
                            a3=int(a3)
                            a1=a1%1000
                            result = [a3,a1,a2, "10^", abs(exp)]                    

            else:
                if coeff < 0:
                    if abs(coeff)<1000:
                        result = ["-",abs(coeff), "10^", exp]
                    else:
                        a1 = abs(coeff)/1000
                        a1=int(a1)
                        a2=abs(coeff)%1000
                        if a1<1000:  
                            result = ["-",a1,a2, "10^", exp]
                        else:
                            a3 = a1/1000
                            a3=int(a3)
                            a1=a1%1000
                            result = ["-",a3,a1,a2, "10^", exp]
                else:
                    if abs(coeff)<1000:
                        result = [coeff, "10^", exp]
                    else:
                        a1 = abs(coeff)/1000
                        a1=int(a1)
                        a2=abs(coeff)%1000
                        if a1<1000:
                            result = [a1,a2, "10^", exp]
                        else:
                            a3 = a1/1000
                            a3=int(a3)
                            a1=a1%1000
                            result = [a3,a1,a2, "10^", exp]
    
    # 处理其他情况
    elif expr == sp.E:
        result.append("E")
    elif expr == sp.pi:
        result.append("pi")
    elif expr == sp.I:
        result.append("i")
    elif expr.is_Symbol:
        result.append(str(expr))
    elif isinstance(expr, sp.Mul):
        # 如果是乘法，添加 '*'，然后递归处理乘数
        result.append('*')
        for arg in expr.args:
            result.extend(to_polish_notation(arg))
    elif isinstance(expr, sp.Add):
        # 如果是加法，添加 '+'，然后递归处理加数
        result.append('+')
        for arg in expr.args:
            result.extend(to_polish_notation(arg))
    elif isinstance(expr, sp.Pow):
        # 如果是幂运算，添加 '**'，然后递归处理底数和指数
        result.append('**')
        result.extend(to_polish_notation(expr.args[0]))
        result.extend(to_polish_notation(expr.args[1]))
    elif expr.func in [sp.sin, sp.cos, sp.tan, sp.log, sp.sqrt, sp.exp]:
        # 如果是一元运算符（如 sin、cos 等）
        result.append(str(expr.func))
        result.extend(to_polish_notation(expr.args[0]))
    else:
        raise ValueError(f"无法识别的表达式: {expr}")
    return result

def save_function_as_polish(expr, filename):
    """
    将表达式以前序遍历的Polish表示法保存到文件中，并加上“SEP”分隔符。
    """
    polish = to_polish_notation(expr)
    with open(filename, 'w') as f:
        f.write("[" + ", ".join(polish) + "]\n")
    print(f"函数已以Polish表示法保存至 {filename}")


def save_dynamical_system_as_polish(dynamical_system, filename):
    """
    将动力系统的每个方程以Polish表示法保存到文件中，并用SEP分隔。
    """
    with open(filename, 'w') as f:
        for eq in dynamical_system:
            polish = to_polish_notation(eq)
            f.write("[" + ", ".join(polish) + "], SEP\n")
    print(f"动力系统已以Polish表示法保存至 {filename}")



for i in range(100000):
    n_vars = random.randint(2, 3)  # 在 2 到 5 之间生成随机整数
    x = sp.symbols(f'x:{n_vars}')
    V, f_system = backward_generation(n_vars)  # 使用随机生成的 n_vars
   
    

    # 以 Polish 表示法保存 V 到 lyapunov_function_polish.txt
    with open("lyapunov_function_polish_poly.txt", 'a') as f:
        
        polish = to_polish_notation(V)  # 假设 to_polish_notation 函数已定义
        polish_str = [str(item) for item in polish]
        f.write("[" + ", ".join(polish_str) + ", ]\n")
        
    
    # 以 Polish 表示法保存 f_system 到 dynamical_system_polish.txt
    with open("dynamical_system_polish_poly.txt", 'a') as f:
        
        f.write("[")
        for eq in f_system:
            polish = to_polish_notation(eq)  # 假设 to_polish_notation 函数已定义
            polish_str = [str(item) for item in polish]
            f.write( ", ".join(polish_str) + ", "+" SEP"+ ", ")
        f.write("]\n")
        
    
    print(f"Iteration {i+1} saved, n_vars = {n_vars}")
    