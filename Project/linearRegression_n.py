import numpy as np
import projectLib as lib
from numpy.linalg import pinv
# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()


#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # ???
    for i in range(trStats["n_ratings"]) :
        x=training[i]
        movieId = x[0]
        userId =x[1]
        userIndex = trStats["n_movies"]+userId
        A[i][movieId]=1.0
        A[i][userIndex]=1
    return A

# we also get c
def getc(rBar, ratings):
    # ???
    c = np.zeros(trStats["n_ratings"])
    for i in range(trStats["n_ratings"]):
        c[i]=ratings[i]-rBar
    return c

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # ???
    b = np.dot(np.dot(pinv(np.dot(A.transpose(), A)), A.transpose()),c)
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    # ???
    b = np.dot(np.dot(pinv(np.dot(A.transpose(), A)+l*np.identity(A.shape[1])), A.transpose()),c)
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p
def neighbourhood(predictions,actual,b,movies,users):
    #get the error matrix
    r = np.full((trStats["n_users"],trStats["n_movies"]),-1000000)
    for user in range(len(r)) :
        print(r[user])
        for movie in range(r[user]) :
            if actual[user][movie]is not Null:
                r = actual[user][movie]-predictions[user][movie]
    #get the distance between movies
    dis = dict()
    for movieA in range(trStats["n_movies"]):
        for movieB in range(trStats["n_movies"]):
            if movieA<movieB:
                numerator = 0.0
                denominator_a = 0.0
                denominator_b = 0.0
                for user in range(trStats["n_users"]):
                    if r[user][movieA] > -100000 and r[user][movieB] > -100000:
                       numerator += r[user][movieA]*r[user][movieB]
                       denominator_a += (math.pow(r[user][movieA],2))
                       denominator_b += (math.pow(r[user][movieB],2))
                if np.abs(numerator/np.sqrt(denominator_a*denominator_b))> 0.499999:
                    dis[(movieA,movieB)] =numerator/np.sqrt(denominator_a*denominator_b)
    return r
    #modify predictions
    # n_predict = len(users)
    # p = np.zeros(n_predict)
    # for user in range(len(r)) :
    #     for movie in range(r[user]) :
    #         if r[user][movie] == -1000000:
    #             rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
    #             if rating > 5: rating = 5.0
    #             if rating < 1: rating = 1.0

b = param(A,c)
pred = predict(trStats['movies'],trStats['users'],rBar,b)
print(neighbourhood(pred,trStats['ratings'],b,trStats['movies'],trStats['users']))


# Unregularised version
#b = param(A, c)

# Regularised version
# for i in range(11):
#     l = i*0.0001
#     b = param_reg(A, c, l)

#     trRMSE = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
#     vlRMSE = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
#     predict(trStats["movies"], trStats["users"], rBar, b)
    #print("Linear regression, l = %f" % l)
    #print("Training loss = %f" % trRMSE)
    #print("Validation loss = %f\n" % vlRMSE)

