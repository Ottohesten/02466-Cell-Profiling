import torch
import time
import os

start_total = time.time()

N = int(1000) 
A = torch.ones([N, N])
B = torch.zeros([N, N])

if torch.cuda.is_available():
	A = A.cuda()
	B = B.cuda()
	compute_hardware = f'(GPU)'
else:
        n = 1
        if  os.environ.get('LSB_DJOB_NUMPROC'):
            n = os.environ.get('LSB_DJOB_NUMPROC')
        torch.set_num_threads(int(n))
        compute_hardware = f'(CPU, num_threads: {n})'

# warm up the system
C = torch.matmul(A, B)

start_matmul = time.time()
reps = 500
for _ in range(reps):
	C = torch.matmul(A, B)
	
end = time.time()
t_matmul = (end-start_matmul)*1e3
t_total = (end-start_total)

print(f'Time to process: \t {t_matmul/reps:.2f} ms/matmul \t ({t_total:.2f} s total) \t {compute_hardware}')
