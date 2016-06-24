import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
import matplotlib.pylab as plt
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
