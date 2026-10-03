import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# 1. Ładowanie danych
df = pd.read_csv('iris.csv')
X = df.drop(['Id', 'Species'], axis=1)
y = df.iloc[:, 5].values # Pobranie gatunków wg indeksu kolumny

# 2. Skalowanie danych
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)


# 6. Wyświetlenie wariancji wyjaśnionej
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
print("LDA Explained Variance Ratio:", lda.explained_variance_ratio_)
# ------------------------------


# --- WYKRES 1: PCA ---
plt.figure(figsize=(10, 6))
for label in pd.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label)
plt.title('PCA: Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# --- WYKRES 2: LDA ---
plt.figure(figsize=(10, 6))
for label in pd.unique(y):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=label)
plt.title('LDA: Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()