import numpy as np
import rbm
import projectLib as lib
import pickle

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5
K_range=(5,10,1)
# SET PARAMETERS HERE!!!
# number of hidden units
F = 8
F_range=(8,15,1)
epochs = 20
epoch_range=(20,500,1)
epsilon = 0.01
epsilon_range=(0.01,0.1,0.01)
B = 10
B_range=(10,50,1)
weightcost = 0.0004
momentum = 0.4


weightcost_range=(0.0002,0.001,0.0001)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


def parameterOptimizer(K_range,F_range,epoch_range,epsilon_range,B_range,weightcost_range):
    lst =[]
    for K in my_range(K_range[0],K_range[1],K_range[2]):
        for F in my_range(F_range[0],F_range[1],F_range[2]):
            for epoch in my_range(epoch_range[0],epoch_range[1],epoch_range[2]):
                for epsilon in my_range(epsilon_range[0],epsilon_range[1],epsilon_range[2]):
                    for B in my_range(B_range[0],B_range[1],B_range[2]):
                        for weightcost in my_range(weightcost_range[0],weightcost_range[1],weightcost_range[2]):
                              params = [K,F,epochs,epsilon,B,weightcost]
                              bestRMSE =mainFunction(K,F,epochs,epsilon,B,weightcost)
                              lst.append(bestRMSE,params)
    optimal_param = min(lst, key = lambda t: t[0])
    return optimal_param











def mainFunction(K,F,epochs,epsilon,B,weightcost):
    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)

    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    grad = np.zeros(W.shape)

    hiddenBiases = np.zeros([1,W.shape[1]])
    visibleBiases = np.zeros([1,W.shape[0]])
    hiddenBiasGrad = np.zeros([1,W.shape[1]])
    visibleBiasGrad = np.zeros([1,W.shape[0]])

    poshidact = np.zeros([1,W.shape[1]])
    neghidact = np.zeros([1,W.shape[1]])
    posvisact = np.zeros([1,W.shape[0]])
    negvisact = np.zeros([1,W.shape[0]])

    bestW = np.zeros(W.shape)
    bestRMSE = 100

    for epoch in range(1, epochs):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
        visitingOrder = np.array_split(visitingOrder, visitingOrder.shape[0] / B)
        # print(visitingOrder)
        for batch in visitingOrder:
            # print(batch)
            temp = np.zeros(W.shape)
            for user in batch:
                # get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, training)
                # build the visible input
                v = rbm.getV(ratingsForUser)
                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]
                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)
                # poshidact = posHiddenProb.transpose()
                # posvisact = np.sum(v,axis=0)
                

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)
                # neghidact = negHiddenProb.transpose()
                # negvisact = np.sum(negData,axis=0)
                # we average over the number of users
                gradientLearningRate = epsilon / epoch
                # print(gradientLearningRate)
                grad = momentum * grad + (1-momentum) * gradientLearningRate * ((posprods - negprods) / trStats['n_users'] - weightcost * np.linalg.norm(temp))
                # hiddenBiasGrad = gradientLearningRate * (poshidact - neghidact) / trStats["n_users"]
                # visibleBiasGrad = gradientLearningRate * (posvisact - negvisact) / trStats["n_users"]
                temp += grad
                # hiddenBiases += hiddenBiasGrad
                # visibleBiases += visibleBiasGrad
            W += temp


        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        if vlRMSE < bestRMSE:
            bestRMSE = vlRMSE
            bestW = W

        print("### EPOCH %d ###" % epoch)
        print("Learning rate = %f" % gradientLearningRate)
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)

    np.save('best_weight.npy', bestW)
    return bestRMSE
parameterOptimizer(K_range,F_range,epoch_range,epsilon_range,B_range,weightcost_range)
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
