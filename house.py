import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from elm import ExtremeLearningMachine  # Memastikan ini mengacu pada kelas yang benar

# Baca data
df = pd.read_csv('house_data.csv')
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

# Pisahkan fitur dan target
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Untuk menggunakan ExtremeLearningMachine, Anda bisa melakukan ini:
elm_model = ExtremeLearningMachine(n_hidden_units=100, n_hidden_layers=1, dropout_prob=0.0, C=100)
elm_model.fit(X_train, y_train)

with open('elm_model.pkl', 'wb') as file:
    pickle.dump(elm_model, file)
