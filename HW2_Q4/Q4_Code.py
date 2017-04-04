# Guo Ziqi - 1000905
# Zhao Juan - 1000918
# Zhang Hao - 1000899

import numpy as np
import matplotlib.pyplot as plt
import pprint
from numpy.linalg import pinv

pp = pprint.PrettyPrinter(indent=1, width=50)

# Q1
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
print("============ Q1 ============")
print("bu:")
print(bu)
print("bi:")
print(bi)



R_hat = []

for user in range(len(R)):
    R_hat_line = []
    for movie in range(len(R[user])):
        R_hat_line.append(r_bar + bu[user] + bi[movie])
    R_hat.append(R_hat_line)

R_hat = np.array(R_hat)

print("R_hat:")
print(R_hat)


# Q2
print("\n============ Q2 ============")
R_error = []
for user in range(len(R)):
    R_error_line = []
    for movie in range(len(R[user])):
        if R[user][movie] == -1:
            R_error_line.append(0)
        else:
            R_error_line.append(R[user][movie] - R_hat[user][movie])

    R_error.append(R_error_line)

print("Error:")
R_error = np.array(R_error)
print(R_error)

d = []

for movie1 in range(len(R[0])):
    d_line = []
    for movie2 in range(len(R[0])):
        # if False:
        if movie1 == movie2:
            d_line.append(0)
        else:
            product = 0.0
            sum1 = 0.0
            sum2 = 0.0
            for user in range(len(R)):
                if R[user][movie1] != -1 and R[user][movie2] != -1:
                    product += R_error[user][movie1] * R_error[user][movie2]
                    sum1 += R_error[user][movie1] ** 2
                    sum2 += R_error[user][movie2] ** 2

            d_line.append(product/(sum1*sum2)**(1.0/2))
            # print(movie1,movie2,product/(sum1*sum2)**(1.0/2))
    d.append(d_line)

# d = np.array(d)
# print("D array:")
# print(d)

R_hat = []

for user in range(len(R)):
    R_hat_line = []
    for movie in range(len(R[user])):
        neighborhood = []
        max = dict()
        for target in range(len(R[user])):
            if movie != target:
                max[target] = d[movie][target]
        max_2 = sorted(max, key=lambda x: abs(max[x]), reverse=True)[:-1]

        for target in max_2:
            # print(user,movie,target,R_error[user][target])
            neighborhood.append(d[movie][target] * R_error[user][target] /
                                (abs(d[movie][max_2[0]]) +
                                 abs(d[movie][max_2[1]])))
        R_hat_line.append(r_bar + bu[user] + bi[movie] + neighborhood[0] + neighborhood[1])
    R_hat.append(R_hat_line)

R_hat = np.array(R_hat)
print("R_hat:")
print(R_hat)

import matplotlib.pyplot as plt
from numpy.linalg import inv

# Q3
print("\n============ Q3 ============")

A = np.array([[1, 0, 2],
              [1, 1, 0],
              [0, 2, 1],
              [2, 1, 1]])
c = np.array([[2], [1], [1], [3]])
AT = A.transpose()
ATA = np.dot(AT,A)
ATA_inv = pinv(ATA)
K = np.dot(AT, c)
print ('q3(a) b value without Regularization:')
print (np.dot(ATA_inv,K))

I = np.identity(3)
lamda = 0
lamda_lst = []
b_lst = []
ab_c_lst = []
b_norm_lst = []
while lamda < 5.1:
    lamda_I = lamda * I
    ATA_inv = pinv(ATA + lamda_I)
    b = np.dot(ATA_inv, K)
    ab_c = np.dot(A, b)-c
    # ab_c_norm = np.linalg.norm(ab_c,ord=2)
    # lamda_b_norm = lamda*np.linalg.norm(b,ord=2)
    ab_c_norm = np.linalg.norm(ab_c,ord = 2)**2
    lamda_b_norm = np.linalg.norm(b,ord = 2)**2
    ab_c_lst.append(ab_c_norm)
    b_norm_lst.append(lamda_b_norm)
    lamda_lst.append(lamda)
    b_lst.append(b)
    lamda += 0.2
print ('\nq3(a) b value with Regularization:')
print (np.array(b_lst))
# print (ab_c_lst)
plt.gca().set_color_cycle(['red',  'blue'])
plt.plot(lamda_lst,ab_c_lst)
plt.plot(lamda_lst,b_norm_lst)
plt.xlabel('lambda')
plt.legend(['2-norm square of Ab-c', '2-norm square of b'], loc='upper left')
plt.show()
















