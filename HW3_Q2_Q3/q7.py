import numpy as np
from numpy.linalg import pinv
import time

H_mat = [[0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1.0 / 3, 0, 1.0 / 3, 0, 1.0 / 3],
         [0, 0, 0.5, 0.5, 0],
         [0, 0, 0, 0, 0]]

H = np.array(H_mat)

# print(H)

N = 5
w = [0, 0, 0, 0, 1]

start_time = time.time()

for theta in [0.1, 0.3, 0.5, 0.85]:
# for theta in [0.85]:
    I = np.identity(N)
    v = np.array([1.0 / N] * 5)
    # pi = np.dot(pinv(I - theta * np.transpose(H)), v.transpose())
    # pi /= np.sum(pi)
    # print(theta, pi)

    pi = np.array([1, 0, 0, 0, 0]).transpose()
    for i in range(100000):
        pi = np.dot(theta * H.transpose(), pi) + v.transpose()
    pi /= np.sum(pi)
    print(pi)

print("Time: ", time.time() - start_time)
