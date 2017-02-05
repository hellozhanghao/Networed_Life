import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
A =np.array([[1,0,2],
[1,1,0],
[0,2,1],
[2,1,1]
])
c = np.array([[2],[1],[1],
[3]])
AT = A.transpose()
ATA = np.dot(AT,A)
ATA_inv = inv(ATA)
K = np.dot(AT, c)
print ('q3(a) b value without Regularization\n')
print (np.dot(ATA_inv,K))
print (ATA)


I = np.identity(3)
lamda = 0
lamda_lst =[]
b_lst = []
ab_c_lst = []
lamda_b_lst = []
while lamda<5.1:
	lamda_I = lamda*I
	ATA_inv = inv(ATA+lamda_I)
	b = np.dot(ATA_inv,K)
	ab_c = np.dot(A,b)-c
	'''ab_c_norm = np.linalg.norm(ab_c,ord=2)
	lamda_b_norm = lamda*np.linalg.norm(b,ord=2)'''
	ab_c_norm = np.linalg.norm(np.dot(ab_c.transpose(),ab_c),ord = 2)
	lamda_b_norm = np.linalg.norm(lamda*np.dot(b.transpose(),b),ord = 2)
	ab_c_lst.append(ab_c_norm)
	lamda_b_lst.append(lamda_b_norm)
	lamda_lst.append(lamda)
	b_lst.append(b)
	lamda +=0.2
'''print ('q3(a) b value with Regularization\n')
print (b_lst)'''
print (ab_c_lst)
plt.gca().set_color_cycle(['red',  'blue'])
plt.plot(lamda_lst,ab_c_lst)
plt.plot(lamda_lst,lamda_b_lst)
plt.legend(['y = Ab-c', 'y = lamda*b'], loc='upper left')
plt.show()



