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

# SET PARAMETERS HERE!!!
# number of hidden units
F = 8
epochs = 20
epsilon = 0.01
B = 10
weightcost = 0.0004

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
            grad = gradientLearningRate * ((posprods - negprods) / trStats['n_users'] - weightcost * np.linalg.norm(temp))
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
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
