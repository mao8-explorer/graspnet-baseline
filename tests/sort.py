import numpy as np
import timeit

np.random.seed(123)

N = 1000
top_num = 10 

arr = np.random.rand(N) < 0.3

def method1():
    indices = np.where(arr)[0][:top_num]
    return indices

def method2():
    indices = arr.argpartition(-top_num,axis=0)[-top_num:]
    return indices

print("Method 1 result: ")
print(method1()) 

print("Method 2 result: ") 
print(method2())

print("Method 1 time: ", timeit.timeit(method1, number=1000))
print("Method 2 time: ", timeit.timeit(method2, number=1000))