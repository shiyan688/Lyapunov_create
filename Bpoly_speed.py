from Bpoly1 import backward_generation,to_polish_notation
import concurrent.futures
import multiprocessing
import sympy as sp

def process_generation(index):
    """
    生成李雅普诺夫函数和动力系统的 Polish 表示法并返回结果。
    """
    n_vars = 2  # 确定变量数量
    
    V, f_system = backward_generation(n_vars)  # 随机生成

    # 生成李雅普诺夫函数的 Polish 表示法
    lyapunov_polish = "[" + ", ".join(map(str, to_polish_notation(V))) + ", ]\n"

    # 生成动力系统的 Polish 表示法
    dynamical_str = "["
    for eq in f_system:
        polish = to_polish_notation(eq)
        dynamical_str += ", ".join(map(str, polish)) + ", " + "SEP" + ", "
    dynamical_str += "]\n"

    return lyapunov_polish, dynamical_str, index

def main():
    lyapunov_buffer = []
    dynamical_system_buffer = []

    # 设置并行执行的进程数
    max_workers = 12

    # 批量生成
    total_iterations = 60
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_generation, i) for i in range(total_iterations)]

        for future in concurrent.futures.as_completed(futures):
            try:
                lyapunov_polish, dynamical_str, index = future.result()
                lyapunov_buffer.append(lyapunov_polish)
                dynamical_system_buffer.append(dynamical_str)

                # 每 50 次写入文件
                if (index + 1) % 5 == 0:
                    with open("lyapunov_function_polish_poly_test.txt", 'a') as f:
                        f.writelines(lyapunov_buffer)
                    with open("dynamical_system_polish_test.txt", 'a') as f:
                        f.writelines(dynamical_system_buffer)
                    lyapunov_buffer = []
                    dynamical_system_buffer = []

                print(f"Iteration {index + 1} completed.")
            except Exception as e:
                print(f"Error in iteration: {e}")

    # 写入剩余内容
    if lyapunov_buffer:
        with open("lyapunov_function_polish_test.txt", 'a') as f:
            f.writelines(lyapunov_buffer)
    if dynamical_system_buffer:
        with open("dynamical_system_test.txt", 'a') as f:
            f.writelines(dynamical_system_buffer)

if __name__ == "__main__":
    main()
