import numpy as np
import matplotlib.pyplot as plt
import pprint
from numpy.linalg import pinv

pp = pprint.PrettyPrinter(indent=1, width=50)

R = np.array([[5, -1, 5, 4],
              [-1, 1, 1, 4],
              [4, 1, 2, 4],
              [3, 4, -1, 3],
              [1, 5, 3, -1]
              ])

c_train = 0
sum_u_i = 0.0

for i in R:
    for j in i:
        if j != -1:
            sum_u_i += j
            c_train += 1

r_bar = sum_u_i / c_train

A = []
c = []
for user in range(len(R)):

    for movie in range(len(R[user])):
        if R[user][movie] == -1:
            continue

        c.append(R[user][movie] - r_bar)

        A_line = []

        for bu in range(len(R)):
            if user == bu:
                A_line.append(1)
            else:
                A_line.append(0)

        for bi in range(len(R[user])):
            if movie == bi:
                A_line.append(1)
            else:
                A_line.append(0)
        A.append(A_line)

# for i in A:
#     print()
#     for j in i:
#         print(str(j), " ", end="")
#
# print()
#
# print(c)

A = np.array(A)
c = np.array(c)

b = np.dot(np.dot(pinv(np.dot(A.transpose(), A)), A.transpose()),c)

bu = b[:-len(R[0])]
bi = b[len(R):]
print(bu)
print(bi)



R_hat = []

for user in range(len(R)):
    R_hat_line = []
    for movie in range(len(R[user])):
        R_hat_line.append(r_bar + bu[user] + bi[movie])
    R_hat.append(R_hat_line)

R_hat = np.array(R_hat)


# part 2
R_error = []
for user in range(len(R)):
    R_error_line = []
    for movie in range(len(R[user])):
        if R[user][movie] == -1:
            R_error_line.append(0)
        else:
            R_error_line.append(R[user][movie] - R_hat[user][movie])

    R_error.append(R_error_line)

R_error = np.array(R_error)
print(R_error)












