#!/usr/bin/python2 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy import stats

#Procesa el dataframe para el uso posterior
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

X = df_scaled.ix[:,:-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']

def mse(matrix):
    #Calcula el mean square error
    return np.mean(np.power(matrix,2))
def mse_comparison(x,y,predictions, coefs):
    #Calcula el mse para ese conjunto de datos
    residuals = predictions - y
    return mse(residuals)
def zscore(x,y,predictions, coefs):

    #Calcula el zscore de la matriz
    print coefs
    v = np.linalg.inv(np.dot(x.T,x))
    vjj = np.diag(v)
    sigma = ( (predictions - y) ** 2).sum()
    sigma = sigma/(x.shape[0] - x.shape[1] - 1)
    z_score = coefs/((np.sqrt(sigma*vjj)))
    print abs(z_score)
    return abs(z_score)
def fss(x, y, x_test, y_test, names_x, comparison_test = mse_comparison, k = 10000):
    p = x.shape[1]-1
    k = min(p, k)
    names_x = np.array(names_x)
    remaining = range(0, p)
    selected = [p]
    current_score = 0.0
    best_new_score = 0.0
    #El error que se obtiene al seleccionar desde 0 hasta i atributos
    training_errors = []
    test_errors = []
    while remaining and len(selected)<=k :
        score_candidates = []
        for candidate in remaining:
            model = lm.LinearRegression(fit_intercept=False)
            indexes = selected + [candidate]
            x_train = x.ix[:,indexes]
            x_test_curr = x_test.ix[:,indexes]
            fitted_model = model.fit(x_train,y)
            predictions_train = model.predict(x_train)
            predictions_validation = model.predict(x_test_curr)
            mse_candidate = comparison_test(x_train,y,predictions_train, model.coef_)
            mse_test = comparison_test(x_test_curr,y_test,predictions_validation, model.coef_)
            score_candidates.append((mse_candidate, mse_test, candidate))
        score_candidates.sort()
        score_candidates[:] = score_candidates[::-1]
        best_new_score, best_test,best_candidate = score_candidates.pop()
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        training_errors.append(best_new_score)
        test_errors.append(best_test)
        print "selected = %s ..."%names_x[best_candidate]
        print "totalvars=%d, mse = %f, mse_test = %f"%(len(indexes),best_new_score, best_test)
    return selected, training_errors, test_errors

names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
selected, training_errors, test_errors = fss(X[istrain],y[istrain],X[istest],y[istest],names_regressors)
plt.plot(np.arange(1,len(training_errors)+1),training_errors)
plt.plot(np.arange(1,len(training_errors)+1),test_errors)
plt.show()
