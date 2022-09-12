import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, train_size=0.8)  # Dividir el set en train y test
    model.fit(X_train, y_train)  # Entrenar modelo
    # Medir R^2 de los resultados con X's test y la 'y' real
    r2 = model.score(X_test, y_test)
    print(f'R^2 Score with {i+1} split: {round(r2,4)}')

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
weight = prediction['Weight'].iloc[0][0]
print(f'Weight of fish 2: {round(weight,4)}')
