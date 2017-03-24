import numpy as np

A = [[0, 0, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]]

A = np.array(A)
C = np.dot(A.transpose(), A)

A2 = np.dot(A, A)
print("A2=\n", A2)

A3 = np.dot(A2, A)
print('A3=\n', A3)

# 8.1

A = [[0, 1, 1, 1, 0],
     [1, 0, 0, 0, 1],
     [1, 0, 0, 1, 1],
     [1, 0, 1, 0, 1],
     [0, 1, 1, 1, 0]]

# degree 3,2,3,3,3

# closeness 0.8,0.2,0.8,0.8,0.8

# eigenvector centrality

eig_value = np.linalg.eig(A)[0]
eig_value = np.abs(eig_value)
print(eig_value)

eig_vector = np.linalg.eig(A)[1].transpose()
print(eig_vector[0])

print(np.dot(A, eig_vector[0]))
# print(eig_value[0] * eig_vector[0])

# 8.3

A = [[0, 1, 1, 0, 0, 0, 1, 0],
     [1, 0, 0, 1, 0, 0, 0, 1],
     [1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 0, 1, 1, 0, 0],
     [0, 0, 1, 1, 0, 1, 1, 0],
     [0, 0, 1, 1, 1, 0, 0, 1],
     [1, 0, 0, 0, 1, 0, 0, 1],
     [0, 1, 0, 0, 0, 1, 1, 0]]

C = np.array([-1, -1, 0, -1, -1, -1, -1, -1])
p = 0.3

prev_C = np.array([-1, -1, -1, -1, -1, -1, -1, -1])

count = 1

while (C != prev_C).any():
    # while True:
    prev_C = np.copy(C)
    copy_C = np.copy(C)
    for healthy in range(len(C)):
        if C[healthy] == -1:
            # find neighbors
            neighbors = []
            for neighbor in range(len(A[healthy])):
                if A[healthy][neighbor] == 1:
                    neighbors.append(neighbor)
            neighbors_infected = 0
            for neighbor in neighbors:
                if C[neighbor] != -1:
                    neighbors_infected += 1
            if neighbors_infected / len(neighbors) >= p:
                copy_C[healthy] = count
                # print(copy_C)
                # print(C)
    count += 1
    C = np.copy(copy_C)

print(C)

# 8.3

S = 0.9
I = 0.1
R = 0.0
beta = 1
gamma = 1 / 3.0
vita = 1 / 50.0

for t in range(200):
    S_pre = S
    I_pre = I
    R_pre = R
    S += -beta * S_pre * I_pre + vita * R_pre
    I += beta * S_pre * I_pre - gamma * I_pre
    R += gamma * I_pre - vita * R_pre
    print(S, I, R)
# 10000 0.33333333333333337 0.03773584905660379 0.6289308176100645



# 8.4
print('\n')
x = np.array([[0, .7, .5, .9, 0],
              [.7, 0, 0, 0, .8],
              [.5, 0, 0, .5, .2],
              [.9, 0, .5, 0, .3],
              [0, .8, .2, .3, 0]])
A = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])
A = A - x
N = len(A)
for i in range(N):
    A[i][i] += sum(x[i])
print(A)

C = np.linalg.inv(A)
print(C)

T = 0
for i in range(N):
    T += C[i][i]

R = []
for i in range(N):
    R.append(sum(C[i]))

IC = []
for i in range(N):
    IC.append(1 / (C[i][i] + (T - 2 * R[i]) / N))

print(IC)
