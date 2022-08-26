import numpy as np
from numba import vectorize
import math
from timeit import default_timer as timer
 
@vectorize(['float32(float32, int32)'], target='cpu')
def with_cpu(x, count):
    for _ in range(count):
        x = math.sin(x)
    return x
 
@vectorize(['float32(float32, int32)'], target='cuda')
def with_cuda(x, count):
    for _ in range(count):
        x = math.sin(x)
    return x
 
data = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
 
for c in [1, 10, 100, 1000, 10000, 20000]:
    print(c)
    cuda_time = 0.0
    cpu_time = 1.0
    for f in [with_cuda, with_cpu]:
        start = timer()
        r = f(data, c)
        elapsed_time = timer() - start
        if f == with_cpu:
            print(f"Time with CPU: {elapsed_time}")
            cpu_time = elapsed_time
        if f == with_cuda:
            print(f"Time with CUDA: {elapsed_time}")
            cuda_time = elapsed_time
    print(f"Acceleration ratio: {cpu_time/cuda_time}")
