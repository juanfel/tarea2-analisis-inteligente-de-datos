import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix 
from scipy.io import mmread 
import sklearn.linear_model as lm 
from sklearn.metrics import mean_squared_error 

def list_duplicates_of(seq,item): 
    start_at = -1 
    locs = [] 
    while True: 
        try: 
            loc = seq.index(item,start_at+1) 
        except ValueError: 
            break 
        else: 
            locs.append(loc) 
            start_at = loc 
    return locs 


######## Pregunta (a) ############################################################


# Se carga el directorio 
data_dir = "./ngrams-deprels-fp1-origin.runtime.budget.numscreen.ratings.seasons.stars/" 

# Se cargan los datos de entrenamiento x en la matriz X
print "Cargando datos en X..." 
X = csr_matrix( mmread(data_dir+'train.x.mm')) 

# Se cargan los datos de entrenamiento y en la matriz y
print "Cargando datos en y..." 
y = np.loadtxt(data_dir+'train.y.dat') 

# Se cargan los datos de test x en la matriz X_test
print "Cargando datos en X_test..." 
X_test = csr_matrix( mmread(data_dir+'test.x.mm')) 

# Se cargan los datos de test y en la matriz y_test
print "Cargando datos en y_test..." 
y_test = np.loadtxt(data_dir+'test.y.dat') 

# Se cargan los datos de validacion de x en la matriz X_val
print "Cargando datos en X_val..." 
X_val = csr_matrix( mmread(data_dir+'dev.x.mm')) 

# Se cargan los datos de validacion de y en la matriz y_val
print "Cargando datos en y_val..." 
y_val = np.loadtxt(data_dir+'dev.y.dat') 

print "Datos cargados"


######## Pregunta (b) ############################################################

# En primer lugar, se prueba con la regresion lineal pero los resultados eran malos

print "Ejecutando lineal..."
model = lm.LinearRegression(fit_intercept = True) 
model.fit (X,y) 

# Se realiza la validacion de los datos
mse = np.mean((model.predict(X_val) - y_val) ** 2) 
print "mse lineal: ", mse
print "R2 lineal: %f"%model.score(X_test, y_test) 

# Se utiliza ridge
print "Ejecutando ridge..."
model = lm.Ridge(alpha=0.3, max_iter=1000, tol=0.001)
# Se ajusta con los datos de entrenamiento
model.fit(X,y) 
mse = np.mean((model.predict(X_val) - y_val) ** 2) 

print "mse ridge: ", mse
print "R2 ridge: %f"%model.score(X_test, y_test) 

# Se utiliza Lasso
print "Ejecutando Lasso..."
model = lm.Lasso(alpha=0.3, max_iter=1000, tol=0.001)
# Se ajusta con los datos de entrenamiento
model.fit(X,y) 
mse = np.mean((model.predict(X_val) - y_val) ** 2) 

print "mse lasso: ", mse
print "R2 lasso: %f"%model.score(X_test, y_test) 
'''
# Se definen los diferentes alphas o lambdas, castigando los coeficientes grandes
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
modelos = [ ] 



# Se itera sobre todos los alphas
for i in range(0,10): 
    print i 
    # Se utiliza ridge
    modelos.append( lm.Ridge(alpha=alphas[i], max_iter=1000, tol=0.001) ) 
    # Se ajusta con los datos de entrenamiento
    modelos[i].fit(X,y) 
    # Se imprimen los coeficientes de determinacion
    print "R2=%f"%modelos[i].score(X, y) 


# Se calcula el error cuadratico medio
mses = [ ] 
for i in range(0,10): 
	# Se guardan en la lista mses los errores para cada alpha
    mses.append = np.mean((modelos[i].predict(X_val) - y_val) ** 2) 
    print("Residual sum of squares N" + str(i) + "{0:.2f}".format(MSEs[i]) ) 

print list_duplicates_of(MSEs, min(MSEs)) 
selected_index = MSEs.index(min(MSEs)) 
print MSEs.index(min(MSEs)) 


print "R2=%f"%models[selected_index].score(X_test, y_test)
'''