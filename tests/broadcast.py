import numpy as np
from timeit import timeit

N = 1000 # 矩阵组个数

# 生成测试数据
R = np.random.random((N,3,3))
A = np.random.random((3,3))

Pc = np.random.random((N,3))
Pw = np.random.random((3,)) 

# 使用for循环
def left_multiply_loop():
    RT = np.empty((N,3,3))
    for i in range(N):
        RT[i,:,:] = np.dot(A, R[i,:,:])
    return RT
    
# 使用广播    
def left_multiply_broadcast():      
    return A @ R


# 使用for循环
def pos_go():
    RT = np.empty((N,3))
    for i in range(N):
        RT[i,:] = np.dot(A, Pc[i,:]) + Pw
    return RT
    

def pos_go_broadcast():
    RT = np.dot(Pc, A.T) + Pw
    return RT


    
# 计算并验证结果    
R1 = left_multiply_loop()
R2 = left_multiply_broadcast()
print(np.allclose(R1, R2)) 


R1 = pos_go()
R2 = pos_go_broadcast()
print(np.allclose(R1, R2)) 


# 测试时间
t1 = timeit(left_multiply_loop, number=1000)  
t2 = timeit(left_multiply_broadcast, number=1000)

print(f'For loop time: {t1}')  
print(f'Broadcast time: {t2}')


# 测试时间
t1 = timeit(pos_go, number=1000)  
t2 = timeit(pos_go_broadcast, number=1000)

print(f'For loop time: {t1}')  
print(f'Broadcast time: {t2}')