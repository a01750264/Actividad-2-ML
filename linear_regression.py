import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Fish.csv')
df = df.drop(columns=['Species'])

# NORMALIZAR LOS DATOS
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df.values)
df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

# DIVIDIR DATASET EN TRAIN Y TEST, Y ENTRENAR EL MODELO
n_splits = 5  # Número de veces que se va a variar el set de training y test
df_X = df.drop(columns=['Weight'])
df_y = df['Weight']

for i in range(n_splits):
    model = SGDRegressor(loss='squared_error', alpha=0.000001, max_iter=10)
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, train_size=0.8)  # Dividir el set en train y test
    model.fit(X_train, y_train)  # Entrenar modelo
    # Medir R^2 de los resultados con X's test y la 'y' real
    y_predict = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    bias = np.mean((y_test - y_predict.mean())**2)
    variance = np.mean(np.var(y_predict))
    print(f'R^2 Score with {i+1} split: {round(r2,4)}')
    print(f'Bias: {round(bias,4)}  Variance: {round(variance,4)}\n')

n = range(len(y_test))
plt.figure()
plt.plot(n, y_predict, 'o-', label='hat y')
plt.plot(n, y_test, 'o-', label='real y', )
plt.legend(loc='best', fancybox=True, shadow=True)
plt.xlabel('n')
plt.ylabel('value')
plt.grid(True)
plt.title('Real y vs hat y')
plt.show()

# PREDICCIONES
X = np.array([0.02, -0.179, 0.821, 0.426, 0.596])
X = pd.DataFrame([X], columns=df_X.columns)
y = model.predict(X)
X.insert(0, 'Weight', y)
prediction = pd.DataFrame(scaler.inverse_transform(X), columns=[X.columns])
weight = prediction['Weight'].iloc[0][0]
print(f'\nWeight of fish 1: {round(weight,4)}')

X = np.array([-0.305, -0.179, 0.871, -0.23, 0.426])
X = pd.DataFrame([X], columns=df_X.columns)
y = model.predict(X)
X.insert(0, 'Weight', y)
prediction = pd.DataFrame(scaler.inverse_transform(X), columns=[X.columns])
weight = abs(prediction['Weight'].iloc[0][0])
print(f'Weight of fish 2: {round(weight,4)}')
