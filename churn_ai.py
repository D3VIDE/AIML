import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
import pandas as pd
import io
import base64

def dataframe_to_html(df, max_rows=10):
    """Convert dataframe to HTML with auto-styled headers"""
    html = df.head(max_rows).to_html(index=False)
    
    # Add Tailwind classes to the table
    html = html.replace('<table', '<table class="min-w-full divide-y divide-gray-200"')
    
    # Add classes to headers
    html = html.replace('<th>', '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">')
    
    # Add classes to cells
    html = html.replace('<td>', '<td class="px-6 py-4 whitespace-nowrap text-md text-gray-500">')
    
    return html

def plot_to_base64(plt):
    """Convert matplotlib plot to base64 image"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def load_data(file_path):
    """Load data from CSV or Excel file"""
    if file_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df_raw = pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung")
    return df_raw


def preprocess_data(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # Imputasi missing values
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # Encoding
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Geography'] = df['Geography'].astype('category')
    df['Geography'] = df['Geography'].cat.codes

    # Simpan salinan sebelum scaling (untuk analisis nanti)
    df_unscaled = df.copy()

    # Scaling numerik (untuk clustering)
    scaled_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = StandardScaler()
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    return df, df_unscaled

def visualisasi_churn(df):
    plots = []
    # 1. Distribusi Churn
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Exited', hue='Exited', palette='Set2', legend=False)
    plt.title("Distribusi Churn")
    plt.xlabel("Exited (0 = Bertahan, 1 = Keluar)")
    plt.ylabel("Jumlah Nasabah")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plots.append(plot_to_base64(plt))

    # 2. Churn berdasarkan Negara
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='Geography', y='Exited', hue='Geography', errorbar=None, palette='muted', legend=False)
    plt.title("Rata-rata Churn berdasarkan Negara")
    plt.xlabel("Geography (Encoded)")
    plt.ylabel("Exit Rate")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plots.append(plot_to_base64(plt))

    return plots
def clustering(df):
    # Ambil data fitur tanpa kolom target
    X = df.drop(columns=['Exited'])
    X_np = X.values

    # Elbow & Silhouette
    distortions = []
    silhouette_scores = []
    K = range(2, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_np)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_np, kmeans.labels_))

    # Plot Elbow & Silhouette
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Distortion')

    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'ro-')
    plt.title('Silhouette Score')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Silhouette Score')

    elbow_silhouette = plot_to_base64(plt)

    return X,elbow_silhouette

def final_clustering(df, X):
    final_k = 4
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df_clustered = df.copy()

    df_clustered['Cluster'] = kmeans.fit_predict(X)

    # Simpan versi final untuk analisis & visualisasi
    
    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X)

    # Plot hasil cluster (PCA)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                    hue=df_clustered['Cluster'], palette='Set2', s=50, alpha=0.8)
    plt.title("Visualisasi Cluster (PCA)")
    plt.xlabel("PCA Komponen 1")
    plt.ylabel("PCA Komponen 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    cluster_plot = plot_to_base64(plt)

    return df_clustered,cluster_plot

def analysis_cluster(df_clustered, df_unscaled):
    df_unscaled['Cluster'] = df_clustered['Cluster']
    plots = []

    # Jumlah anggota tiap cluster
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_unscaled, x='Cluster',hue='Cluster', palette='Set3', legend=False)
    plt.title("Jumlah Nasabah per Cluster")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plots.append(plot_to_base64(plt))
    # Rata-rata fitur penting per cluster
    important_features = ['Age', 'Balance', 'EstimatedSalary', 'Exited']

    plt.figure(figsize=(12, 6))
    for i, feature in enumerate(important_features):
        plt.subplot(2, 2, i + 1)
        sns.barplot(data=df_unscaled, x='Cluster', y=feature, hue='Cluster', palette='pastel', errorbar=None, legend=False)
        plt.title(f'Rata-rata {feature} per Cluster')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xlabel("Cluster")
        plt.ylabel(feature)

    feature_plots = plot_to_base64(plt)
    plots.append(feature_plots)

    return df_unscaled,plots