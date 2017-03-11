import numpy as np
import rbm
v = np.array([
    [1,2,3,4],
    [5,6,7,8]
])
w = np.array([
    [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    [[4,3,2,1],[8,7,6,5],[12,11,10,9]]
])

print(rbm.visibleToHiddenVec(v,w))

