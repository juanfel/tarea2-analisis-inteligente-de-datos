import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pylab as plt
from sklearn import cross_validation
#Funciones
def regularizate(Xtrain,ytrain,names_regressors,model,alphas, title):
    #Regulariza de acuerdo a la función modelo y a los alphas dados
    coefs = []
    for a in alphas:
        model.set_params(alpha=a)
        model.fit(Xtrain, ytrain)
        coefs.append(model.coef_)
    ax = plt.gca()
    for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
        print alphas.shape
        print y_arr.shape
        plt.plot(alphas, y_arr, label=label)
    plt.legend()
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Regularization Path ' + title)
    plt.axis('tight')
    plt.legend(loc=2)
    plt.show()

def geterrors(Xtrain,ytrain,Xtest,ytest,names_regressors,model,alphas_,title):
    #Obtiene los gráficos de los errores de test y entrenamiento para el
    #modelo dado
    coefs = []
    mse_test = []
    mse_train = []
    for a in alphas_:
        model.set_params(alpha=a)
        model.fit(Xtrain, ytrain)
        yhat_train = model.predict(Xtrain)
        yhat_test = model.predict(Xtest)
        mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
        mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
    ax = plt.gca()
    ax.plot(alphas_,mse_train,label='train error ' + title)
    ax.plot(alphas_,mse_test,label='test error ' + title)
    plt.legend(loc=2)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()

def MSE(y,yhat): return np.mean(np.power(y-yhat,2))
def kfoldcrossval(Xtrain,ytrain,names_regressors,model,alphas,title):
    #Hace la validación cruzada del modelo
    Xm = Xtrain.as_matrix()
    ym = ytrain.as_matrix()
    k_fold = cross_validation.KFold(len(Xm),10)
    best_cv_mse = float("inf")
    for a in alphas_:
        model.set_params(alpha=a)
        mse_list_k10 = [MSE(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val]) for train, val in k_fold]
        if np.mean(mse_list_k10) < best_cv_mse:
            best_cv_mse = np.mean(mse_list_k10)
            best_alpha = a
            print "BEST PARAMETER %s =%f, MSE(CV)=%f"%(title,best_alpha,best_cv_mse)
#Seteo de datos
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

X = X.drop('intercept', axis=1)
Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]

#Pregunta a
alphas_ = np.logspace(4,-1,base=10)
coefs = []
model = Ridge(fit_intercept=True,solver='svd')
regularizate(Xtrain,ytrain,names_regressors,model,alphas_,"RIDGE")

#Pregunta b
alphas_ = np.logspace(1,-2,base=10)
clf = Lasso(fit_intercept=True)
regularizate(Xtrain,ytrain,names_regressors,clf,alphas_,"LASSO")

#Pregunta c
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(2,-2,base=10)
model = Ridge(fit_intercept=True)
geterrors(Xtrain,ytrain,Xtest,ytest,names_regressors,model,alphas_,"RIDGE")

#Pregunta d
alphas_ = np.logspace(1,-2,base=10)
clf = Lasso(fit_intercept=True)
geterrors(Xtrain,ytrain,Xtest,ytest,names_regressors,clf,alphas_,"RIDGE")

#Pregunta e
model = Ridge(fit_intercept=True)
alphas_ = np.logspace(2,-2,base=10)
kfoldcrossval(Xtrain,ytrain,names_regressors,model,alphas_,"RIDGE")

clf = Lasso(fit_intercept=True)
alphas_ = np.logspace(1,-2,base=10)
kfoldcrossval(Xtrain,ytrain,names_regressors,model,alphas_,"LASSO")
