import pprint

pp = pprint.PrettyPrinter()

G = [[1.0, 0.5, 0.5],
     [0.5, 1.0, 0.5],
     [0.5, 0.5, 1.0]]
r = [1.0, 2.0, 1.0]


# G = [[1.0, 0.1, 0.2,0.3],
#      [0.2, 1.0, 0.1, 0.1],
#      [0.2, 0.1, 1.0, 0.1],
#      [0.1, 0.1, 0.3, 1.0]]
# r = [2.0, 2.5, 1.5, 2.0]

n = 0.1

pi = [[1.0, 1.0, 1.0]]
SIRi = []


# Gji = send from i to j

def get_p():
    return pi[len(pi) - 1]


def get_SIR():
    return SIRi[len(SIRi) - 1]


def calc_SIR(i):
    p = get_p()
    sum = 0.0
    # print(sum)
    for j in range(len(G)):
        if i == j: continue
        sum += (G[i][j] * p[j])

    return (G[i][i] * p[i]) / (sum + n)


def update_SIRi():
    SIR = []
    for i in range(len(G)):
        SIR.append(calc_SIR(i))
    SIRi.append(SIR)

def update_pi():
    p_next = []
    p_old = get_p()
    SIR = get_SIR()
    for i in range(len(G)):
        p_next.append((r[i]/SIR[i])*p_old[i])
    pi.append(p_next)

update_SIRi()



for k in range(10):
    update_pi()
    update_SIRi()

print("SIR")
pp.pprint(SIRi)
print("p")
pp.pprint(pi)
