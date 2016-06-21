import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy import stats

url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)

# Esta linea elimina la columna "unnamed: 0", ya que esta contiene la  
# numeracion de la fila y no datos relevantes para el analisis
df = df.drop('Unnamed: 0', axis=1)

# Se guarda en la variable istrain_str los valores que toma la columna "train"
istrain_str = df['train']

# En un arreglo se guarda un valor "True" cuando el dato es de entrenamiento (T)
# y en otro caso, dato de test (F) toma el valor de "False"
istrain = np.asarray([True if s=='T' else False for s in istrain_str])

# Aplica el contrario a cada elemento del arreglo "istrain", por lo que en
# "istest" se marca como "True" a los datos que son de test, y False los demas
istest = np.logical_not(istrain)

# Elimina la columna train del dataframe, ya con los valores guardados en los
# arreglos anteriores
df = df.drop('train', axis=1)

# Informa la dimension del dataframe de la forma (num de filas, num columnas)
#print df.shape

# Se muestra informacion de las columnas como son la cantidad de filas, si la
# columna tiene valores nulos y el tipo de dato que contiene.
#print df.info()

# Se muestra una completa descripcion de los datos, como son la media, 
# el valor minimo, el valor maximo, los percentiles 1, 2 y 3, y otros valores
#print df.describe()

# Se instancia 
scaler = StandardScaler()
# Se crea un nuevo dataframe df_scaled con los datos normalizados
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

# Se saca la ultima columna lpsa
X = df_scaled.ix[:,:-1]
# Se guarda el tamano de las filas del arreglo X en N
N = X.shape[0]
# Al dataframe X se le agrega una nueva columna que representa el intercepto
# llenado con unos
X.insert(X.shape[1], 'intercept', np.ones(N))
# Se guarda la columna lpsa en y, que seria el valor que se desea predecir
y = df_scaled['lpsa']

# En Xtrain e ytrain se guardan los valores del dataframe X y del vector y 
# que son datos de entrenamiento, respectivamente
Xtrain = X[istrain]
ytrain = y[istrain]

# En Xtest e ytest se guardan los valores del dataframe x y del vector y 
# que son datos de test, respectivamente
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]

# Se instancia una regresion lineal. Se le manda como parametro 
# fit_intercept=False, por lo que la intercepcion no se considerara en
# los calculos
linreg = lm.LinearRegression(fit_intercept = False)

# Se ajusta la regresion con los datos de entrenamiento
result = linreg.fit(Xtrain, ytrain)

#zsc = (X - X.mean())/X.std()
Z = stats.zscore(X['lcavol'])


#print zsc['lcavol']
#print linreg.coef_
#print("Residual sum of squares: %.2f" % np.mean((linreg.predict(Xtest) - ytest) ** 2))
#print('Varianza explicada: %.2f\n' % linreg.score(Xtest, ytest))

yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(yhat_test - ytest, 2))
from sklearn import cross_validation
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
k_fold = cross_validation.KFold(len(Xm),10)
mse_cv = 0
for k, (train, val) in enumerate(k_fold):
	linreg = lm.LinearRegression(fit_intercept = False)
	linreg.fit(Xm[train], ym[train])
	yhat_val = linreg.predict(Xm[val])
	mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
	mse_cv += mse_fold
mse_cv = mse_cv / 10