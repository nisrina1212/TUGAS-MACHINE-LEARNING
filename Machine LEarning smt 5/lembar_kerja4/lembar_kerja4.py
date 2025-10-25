import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Pastikan file CSV tersedia ---
if not os.path.exists("kelulusan.csv"):
    raise FileNotFoundError("File 'kelulusan.csv' tidak ditemukan di folder kerja!")

# --- 2. Baca dataset ---
df = pd.read_csv("kelulusan.csv")
print("=== Informasi Data ===")
print(df.info())
print("\n=== 5 Data Pertama ===")
print(df.head())

# --- 3. Cek missing value dan hapus duplikat ---
print("\n=== Jumlah Missing Value ===")
print(df.isnull().sum())
df = df.drop_duplicates()

# --- 4. Visualisasi dasar ---
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()

print("\n=== Statistik Deskriptif ===")
print(df.describe())

plt.figure(figsize=(6, 4))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

if 'Waktu_Belajar_Jam' in df.columns and 'Lulus' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
    plt.title("Hubungan IPK vs Waktu Belajar")
    plt.show()

# --- 5. Heatmap korelasi (khusus kolom numerik) ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Variabel Numerik")
plt.show()

# --- 6. Fitur baru ---
if 'Jumlah_Absensi' in df.columns:
    df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
if 'Waktu_Belajar_Jam' in df.columns:
    df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan hasil preprocessing
df.to_csv("processed_kelulusan.csv", index=False)
print("\nData berhasil disimpan ke 'processed_kelulusan.csv'")

# --- 7. Split data untuk Machine Learning ---
from sklearn.model_selection import train_test_split

if 'Lulus' not in df.columns:
    raise ValueError("Kolom target 'Lulus' tidak ditemukan!")

X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\n=== Ukuran Dataset ===")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)
