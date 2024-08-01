import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())
print(train_df.info())
print(train_df.describe())
print(train_df.columns)
print(train_df.isnull().sum())

# Analisis distribusi target
sns.countplot(x='Survived', data=train_df)
plt.title('Distribusi Target Survived')
plt.show()

# Analisis distribusi fitur
sns.histplot(train_df['Age'].dropna(), kde=True)
plt.title('Distribusi Umur')
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=train_df)
plt.title('Distribusi Fare Berdasarkan Kelas')
plt.show()

# Imputasi untuk kolom numerik
numeric_features = ['Age','Fare']
numeric_transformer = SimpleImputer(strategy='median')

# Imputasi untuk kolom kategorikal
categorical_features = ['Embarked','Sex']
categorical_transformer = SimpleImputer(strategy='most_frequent')

# Imputasi nilai yang hilang
train_df['Age'] = numeric_transformer.fit_transform(train_df[['Age']])

# Membuat fitur baru
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# Mengkodekan variabel kategorikal
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Sex'], drop_first=True)

# Menskalakan fitur numerik
scaler = StandardScaler()
train_df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(train_df[['Age', 'Fare', 'FamilySize']])

# Memisahkan fitur dan target
X = train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train_df['Survived']
print(X.head())
# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Regresi Logistik
log_reg = LogisticRegression(max_iter=1000)

# Melatih model dengan validasi silang
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')

# Tampilkan hasil validasi silang
print(f'Validasi Silang Akurasi: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Membuat model Hutan Acak
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(y_pred)

# Latih model pada data pelatihan
log_reg.fit(X_train, y_train)

# Prediksi pada data pengujian
y_pred = log_reg.predict(X_test)

# Evaluasi model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Menyimpan model Hutan Acak
joblib.dump(rf, 'best_random_forest_model.pkl')

# Memuat model (ketika diperlukan)
loaded_model = joblib.load('best_random_forest_model.pkl')