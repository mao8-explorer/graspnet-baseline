import numpy as np
import time

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm

def euclidean_distance(q1, q2):
    distance = np.linalg.norm(q1 - q2)
    return distance

def angle_distance(q1, q2):
    dot_product = np.dot(q1, q2)
    squared_distance = 2 * (dot_product**2) - 1
    distance = np.arccos(squared_distance)
    return distance

def slerp_distance(q1, q2):
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1, 1)
    angle = np.arccos(dot_product)
    distance = angle * np.linalg.norm(q2 - q1)
    return distance

# 生成随机四元数数据
np.random.seed(42)  # 设置随机种子，保证结果可重现
num_samples = 50
q_origins = np.random.rand(num_samples, 4) - 0.5
q_homes = np.random.rand(num_samples, 4) - 0.5

# 归一化处理
q_origins = np.array([normalize_quaternion(q) for q in q_origins])
q_homes = np.array([normalize_quaternion(q) for q in q_homes])

# 生成符合逻辑关系的 q_flip
q_flips = np.array([-q[1], q[0], q[3], -q[2]] for q in q_origins)

# 进行对比实验
methods = ["Euclidean Distance", "Angle Distance", "SLERP Distance"]

consistent_counts = {method: 0 for method in methods}

for i in range(num_samples):
    print("Data Set", i+1)
    print("q_origin:", q_origins[i])
    print("q_flip:", q_flips[i])
    print("q_home:", q_homes[i])
    print("-----------------------------------------")

    closest_q_origin = None
    closest_q_flip = None
    closest_methods = []

    min_distances = {method: float("inf") for method in methods}

    for method in methods:
        start_time = time.time()

        if method == "Euclidean Distance":
            distance_origin = euclidean_distance(q_homes[i], q_origins[i])
            distance_flip = euclidean_distance(q_homes[i], q_flips[i])
        elif method == "Angle Distance":
            distance_origin = angle_distance(q_homes[i], q_origins[i])
            distance_flip = angle_distance(q_homes[i], q_flips[i])
        elif method == "SLERP Distance":
            distance_origin = slerp_distance(q_homes[i], q_origins[i])
            distance_flip = slerp_distance(q_homes[i], q_flips[i])