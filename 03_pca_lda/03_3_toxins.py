import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# 1. Ładowanie danych
df_toxins = pd.read_csv('ToxinsbaseClass.csv')

# Przygotowanie cech: usuwamy kolumnę celu 'Toxicity' oraz 'Latitude '
X = df_toxins.drop(['Toxicity', 'Latitude '], axis=1).values
y = df_toxins['Toxicity'].values

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. Standaryzacja
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. PCA - redukcja do 2 składowych
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)

# 5. Model Regresji Logistycznej
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_pca, y_train)

# 6. Predykcja i statystyki
y_pred = classifier.predict(X_test_pca)
print("\nMacierz pomyłek (Toxins):")
print(confusion_matrix(y_test, y_pred))
print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")


# 7. Wizualizacja obszarów decyzyjnych
def plot_decision_regions(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # Naprawa błędu z poprzedniego zadania (użycie color zamiast c)
    for i, j in enumerate(np.unique(y_set)):
        color_val = np.array(ListedColormap(('red', 'green', 'blue'))(i)).reshape(1, -1)
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=color_val, label=f'Klasa {j}', edgecolors='black')

    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()


plot_decision_regions(X_train_pca, y_train, 'Logistic Regression (Toxins Training set)')
plot_decision_regions(X_test_pca, y_test, 'Logistic Regression (Toxins Test set)')