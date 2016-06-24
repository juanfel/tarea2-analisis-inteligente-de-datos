import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error 

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
    z_score = coefs/np.sqrt((mse_comparison(x,y,predictions,coefs)*vjj)/(x.shape[0] - x.shape[1]))
    return z_score

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


# Funcion que calcula la desviacion estandar
def desvest(mse, n, j):
	desv_est = np.sqrt(mse * n / (n - j - 1))
	return desv_est
# Funcion que calcula las diagonales de la matriz (xTx)-1
def diagval(xtrain):	
	diag_val = np.diag(np.linalg.pinv(np.dot(xtrain.T, xtrain))) 
	return diag_val
# Funcion que calcula el zscore
def zzscore(coef, desv_est, diagonal):
	z = np.divide(coef, np.multiply(desv_est, np.sqrt(diagonal))) 
	return z


# Se obtiene el valor de y estimado para los datos de entrenamiento
yptrain = linreg.predict(Xtrain) 
# Se calcula el error cuadratico medio de los datos de entrenamiento
msetrain = mean_squared_error(ytrain, yptrain) 
# Para realizar las iteraciones, se guarda en n la cantidad de datos, en este caso,
# son 67 datos de entrenamiento
n = ytrain.shape[0]
# Se guarda en j la cantidad de variables, en este caso son 9
j = Xtrain.shape[1] 

# Se calcula la desviacion estandar
desv_est = desvest(msetrain, n, j)
# Se obtiene la diagonal de los datos de entrenamiento
diag_val = diagval(Xtrain)
# Se calculan los zscore
Zscores = zzscore(linreg.coef_, desv_est, diag_val)
print Zscores

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