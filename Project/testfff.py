import numpy as np
import rbm
v = np.array([
    [1,0,0,0],
    [0,0,0,1]
])
w = np.array([
    [[1,0,0,0],[0,1,0,8],[9,10,11,12]],
    [[4,3,2,1],[8,7,6,5],[12,11,10,9]]
])

print(rbm.visibleToHiddenVec(v,w))


h = rbm.visibleToHiddenVec(v,w)
print(rbm.hiddenToVisible(h,w))
print(np.sum(rbm.hiddenToVisible(h,w)))