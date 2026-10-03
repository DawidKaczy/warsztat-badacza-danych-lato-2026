# Importowanie niezbędnych bibliotek
import pandas as pd # Do pracy z tabelami danych
import numpy as np # Do operacji matematycznych
import matplotlib.pyplot as plt # Do rysowania wykresów
import seaborn as seabornInstance # Do ładniejszych wykresów statystycznych
from sklearn.model_selection import train_test_split # Do podziału danych na zbiór treningowy i testowy
from sklearn.linear_model import LinearRegression # Moduł algorytmu regresji liniowej
from sklearn import metrics # Do obliczania błędów (oceny modelu)

# Wczytanie danych z pliku CSV (oddzielonych średnikiem)
dataset = pd.read_csv('Kruk_VLagun_156.csv', sep=';')

# Sprawdzenie wielkości tabeli i podstawowych statystyk
print("Wymiary danych:", dataset.shape)
print("\n")
print("Podsumowanie statystyczne:", dataset.describe())

# Zamiana formatu kolumn na liczby (coerce zamienia błędy, np. "18.mar", na wartości puste NaN)
dataset['SS'] = pd.to_numeric(dataset['SS'], errors='coerce')
dataset['SecchDisc'] = pd.to_numeric(dataset['SecchDisc'], errors='coerce')

# --- WIZUALIZACJA DANYCH ---

# 1. Wykreślenie punktów danych na wykresie 2D (wykres rozrzutu)
x1 = dataset['SS']
y1 = dataset['SecchDisc']
plt.figure(figsize=(8,6))
plt.scatter(x1, y1) # Tworzy chmurę punktów
plt.title('SecchiDisc vs SS')
plt.xlabel('SS') # Oś X: Zawiesina
plt.ylabel('SecchDisc') # Oś Y: Widzialność krążka
plt.show()

# 2. Rysowanie wykresu rozkładu (histogramu) dla widzialności
plt.figure()
seabornInstance.histplot(dataset['SecchDisc'], kde=True, color='blue') # kde=True dodaje krzywą gęstości
plt.title('Rozkład wartości dla SecchDisc')
plt.xlabel('SecchDisc')
plt.ylabel('Częstotliwość')
plt.show()

# --- PRZYGOTOWANIE DO UCZENIA MASZYNOWEGO ---

# Definicja atrybutów (X) i etykiet (y). 
# reshape(-1,1) jest konieczne, by przekształcić jednowymiarową listę w tablicę 2D (wymóg scikit-learn)
X = dataset['SS'].values.reshape(-1,1)
y = dataset['SecchDisc'].values.reshape(-1,1)

# Podział danych na zbiór do uczenia (80%) i do testowania (20%)
# random_state=0 zapewnia, że przy każdym uruchomieniu podział będzie taki sam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Utworzenie modelu Regresji Liniowej
regressor = LinearRegression()

# Uczenie modelu na danych treningowych (dopasowywanie linii do punktów)
regressor.fit(X_train, y_train)

# Wyświetlenie matematycznych parametrów narysowanej linii (y = ax + b)
print("Wartość przechwycenia (punkt przecięcia z osią Y, 'b'):")
print(regressor.intercept_)
print("\n")
print("Wartość nachylenia (współczynnik 'a'):")
print(regressor.coef_)

# --- TESTOWANIE I EWALUACJA MODELU ---

# Model próbuje przewidzieć widzialność (y) na podstawie testowych wartości zawiesiny (X_test)
y_pred = regressor.predict(X_test)

# Stworzenie tabeli porównującej to, co było w rzeczywistości, z tym co przewidział model
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

# Rysowanie wykresu słupkowego dla pierwszych 25 przewidywań (Rzeczywiste vs Przewidywane)
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Rysowanie ostatecznego wykresu z czerwoną linią trendu nałożoną na rzeczywiste punkty testowe
plt.scatter(X_test, y_test,  color='gray') # Prawdziwe punkty na szaro
plt.plot(X_test, y_pred, color='red', linewidth=2) # Linia przewidywań modelu na czerwono
plt.show()

# --- OBLICZANIE BŁĘDÓW ---

# Wypisanie metryk świadczących o jakości modelu
print('Mean Absolute Error (Średni błąd bezwzględny):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (Błąd średniokwadratowy):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (Pierwiastek błędu średniokwadratowego):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Wartość nachylenia (Slope / współczynnik kierunkowy): -0.00201303
# Znak minus (-) oznacza korelację ujemną (odwrotnie proporcjonalną).
# Zgadza się to z logiką: im więcej zawiesiny (SS) w wodzie, tym mniejsza jest przejrzystość (SecchDisc).
# Sama wartość -0.002 mówi nam, że wzrost stężenia zawiesiny o 1
# jednostkę powoduje spadek widzialności krążka Secchiego o około 0.002 metra (czyli 2 milimetry).