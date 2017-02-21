import numpy as np
from numpy.linalg import pinv
import time


theta = 0.85

start_time = time.time()
# a
print("---------A")


H_mat = [[1, 0],
         [1.0 / 3, 2.0 / 3]]

H = np.array(H_mat)
N = 2
I = np.identity(N)
v = np.array([1.0 / N] * N)
# pi = np.dot(pinv(I - theta * np.transpose(H)), v.transpose())
# pi /= np.sum(pi)
# print(pi)

pi = np.array([0.5, 0.5]).transpose()
for i in range(100000):
    pi = np.dot(theta * H.transpose(), pi) + v.transpose()
pi /= np.sum(pi)
# print(theta, pi)
pi_A = pi[0]
pi_B = pi[1]

print("pi_A, pi_B = ",pi)


# b
print("---------B")

theta = 0.85

H_mat = [[0, 1],
         [1, 0]]

H = np.array(H_mat)
N = 2
pi = np.array([1.0 / N] * N)

I = np.identity(N)
v = np.array([1.0 / N] * N)
for i in range(100000):
    pi = np.dot(theta * H.transpose(), pi) + v.transpose()
pi /= np.sum(pi)

pi_1_2 = pi

print("pi_1, pi_2 = ",pi_1_2)


H_mat = [[1.0/2, 0, 1.0/2],
         [1.0/2, 0, 1.0/2],
         [0,0,0]]

H = np.array(H_mat)
N = 3
pi = np.array([1.0 / N] * N)
I = np.identity(N)
v = np.array([1.0 / N] * N)
for i in range(100000):
    pi = np.dot(theta * H.transpose(), pi) + v.transpose()
pi /= np.sum(pi)

pi_3_4_5 = pi

print("pi_3, pi_4, pi_5 = ",pi_3_4_5)


# c
print("---------C")
pi = np.concatenate((pi_A * pi_1_2, pi_B * pi_3_4_5), axis=0)

print(pi)
print("Time: ", time.time()-start_time)
