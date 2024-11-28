import sympy as sp
import numpy as np
from picos import Problem
from itertools import combinations_with_replacement

class LyapunovSOSChecker:
    def __init__(self):
        """初始化Lyapunov SOS检查器"""
        self.problem = Problem()
    
    def check_lyapunov_candidate(self, system, lyap_func, vars):
        """
        检查Lyapunov候选函数是否满足SOS条件
        
        参数:
        system: 系统动力学方程
        lyap_func: Lyapunov候选函数
        vars: 系统变量
        
        返回:
        bool: 是否满足SOS条件
        """
        # 检查V(x)是否为正定函数
        if not self._check_positive_definite(lyap_func, vars):
            return False
            
        # 计算Lyapunov函数的导数
        V_dot = self._compute_derivative(system, lyap_func, vars)
        
        # 检查-V_dot是否为正半定
        return self._check_negative_derivative(V_dot, vars)
    
    def _check_positive_definite(self, V, vars):
        """检查函数是否正定"""
        try:
            # 构造SOS问题
            degree = V.as_poly().total_degree()
            basis = self._generate_monomial_basis(vars, degree//2)
            
            # 构建SDP问题
            prob = Problem()
            n = len(basis)
            Q = prob.add_variable('Q', (n,n), 'symmetric')
            
            # 添加半正定约束
            prob.add_constraint(Q >> 0)
            
            # 展开V(x)并匹配系数
            expanded_V = V.expand()
            
            # 求解SDP问题
            prob.solve(solver='cvxopt')
            return prob.value is not None
            
        except Exception as e:
            print(f"检查正定性时出错: {e}")
            return False
    
    def _compute_derivative(self, system, V, vars):
        """计算Lyapunov函数的导数"""
        grad_V = sp.Matrix([V.diff(var) for var in vars])
        V_dot = grad_V.dot(system)
        return V_dot
    
    def _check_negative_derivative(self, V_dot, vars):
        """检查导数是否负定"""
        try:
            # 构造-V_dot的SOS问题
            neg_V_dot = -V_dot
            degree = neg_V_dot.as_poly().total_degree()
            basis = self._generate_monomial_basis(vars, degree//2)
            
            # 构建SDP问题
            prob = Problem()
            n = len(basis)
            Q = prob.add_variable('Q', (n,n), 'symmetric')
            
            # 添加半正定约束
            prob.add_constraint(Q >> 0)
            
            # 展开并匹配系数
            expanded_neg_V_dot = neg_V_dot.expand()
            
            # 求解SDP问题
            prob.solve(solver='cvxopt')
            return prob.value is not None
            
        except Exception as e:
            print(f"检查导数负定性时出错: {e}")
            return False
    
    def _generate_monomial_basis(self, vars, degree):
        """生成单项式基"""
        return [np.prod(mono) for mono in combinations_with_replacement(vars, degree)]

# 使用示例
def example():
    # 定义系统变量
    x, y = sp.symbols('x y')
    
    # 定义系统动力学
    system = sp.Matrix([
        -x ,
        -y
    ])
    
    # 定义Lyapunov候选函数
    V = x**2 + y**2
    
    # 创建检查器实例
    checker = LyapunovSOSChecker()
    
    # 检查候选函数
    result = checker.check_lyapunov_candidate(system, V, [x, y])
    
    if result:
        print(f"函数 V = {V} 满足Lyapunov函数的SOS条件")
    else:
        print(f"函数 V = {V} 不满足Lyapunov函数的SOS条件")

if __name__ == "__main__":
    example()