import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
def dataframe_to_html(df, max_rows=10):
    """Convert dataframe to HTML with auto-styled headers"""
    html = df.head(max_rows).to_html(index=False)
    html = html.replace('<table', '<table class="min-w-full divide-y divide-gray-200"')
    html = html.replace('<th>', '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">')
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

    return X, elbow_silhouette

def final_clustering(df, X):
    final_k = 4
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X)

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

    return df_clustered, cluster_plot

def analysis_cluster(df_clustered, df_unscaled):
    df_unscaled['Cluster'] = df_clustered['Cluster']
    plots = []

    # Jumlah anggota tiap cluster
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_unscaled, x='Cluster', hue='Cluster', palette='Set3', legend=False)
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

    return df_unscaled, plots

def genetic_algorithm(df):
    # Siapkan data X dan y (pastikan sudah preprocessing & scaling)
    X = df.drop(columns=['Exited', 'Cluster']).values
    y = df['Exited'].values
    feature_names = df.drop(columns=['Exited']).columns

    # GA setup
    def eval_individual(individual):
        if sum(individual) == 0:
            return 0.0,
        X_selected = X[:, np.array(individual, dtype=bool)]
        clf = KNeighborsClassifier(n_neighbors=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy').mean(),

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=30) #ini 
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20 ,    
                        stats=tools.Statistics(lambda ind: ind.fitness.values),
                        halloffame=hof, verbose=True)  #ini

    # Tampilkan hasil terbaik
    best_ind = hof[0]
    selected = [name for flag, name in zip(best_ind, feature_names) if flag == 1]
    print(f"Fitur terpilih: {selected}")
    return selected

def knn_comparison(df, selected_features):
    # Target
    y = df['Exited']

    # Fitur lengkap (semua fitur kecuali target)
    X_full = df.drop(columns=['Exited'])

    # Fitur hasil seleksi GA
    X_ga = df[selected_features]

    # KNN dengan semua fitur
    knn_full = KNeighborsClassifier(n_neighbors=5)
    acc_full = cross_val_score(knn_full, X_full, y, cv=5, scoring='accuracy').mean()

    # KNN dengan fitur hasil GA
    knn_ga = KNeighborsClassifier(n_neighbors=5)
    acc_ga = cross_val_score(knn_ga, X_ga, y, cv=5, scoring='accuracy').mean()

    # Siapkan data
    labels = ['KNN (Semua Fitur)', 'KNN (Fitur GA)']
    scores = [acc_full, acc_ga]

    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, scores, color=['skyblue', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title("Perbandingan Akurasi KNN")
    plt.ylabel("Akurasi")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Tambahkan nilai akurasi di atas bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plot_img = plot_to_base64(plt)
    
    return knn_ga, plot_img, {'KNN (Semua Fitur)': acc_full, 'KNN (Fitur GA)': acc_ga}

def confusion_matrix_comparison(df, selected_features):
    # 1. Dataset
    X = df.drop(columns=['Exited'])     # Semua fitur
    y = df['Exited']                    # Target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Model dengan semua fitur
    knn_full = KNeighborsClassifier(n_neighbors=5)
    knn_full.fit(X_train, y_train)
    y_pred_full = knn_full.predict(X_test)

    # 4. Model dengan fitur hasil GA
    X_train_ga = X_train[selected_features]
    X_test_ga = X_test[selected_features]

    knn_ga = KNeighborsClassifier(n_neighbors=5)
    knn_ga.fit(X_train_ga, y_train)
    y_pred_ga = knn_ga.predict(X_test_ga)

    # 5. Evaluasi metrics
    full_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_full),
        "Precision": precision_score(y_test, y_pred_full),
        "Recall": recall_score(y_test, y_pred_full),
        "F1 Score": f1_score(y_test, y_pred_full)
    }
    
    ga_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_ga),
        "Precision": precision_score(y_test, y_pred_ga),
        "Recall": recall_score(y_test, y_pred_ga),
        "F1 Score": f1_score(y_test, y_pred_ga)
    }

    # 6. Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_full = confusion_matrix(y_test, y_pred_full)
    disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=['Bertahan', 'Keluar'])
    disp_full.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title("Confusion Matrix - Semua Fitur")

    cm_ga = confusion_matrix(y_test, y_pred_ga)
    disp_ga = ConfusionMatrixDisplay(confusion_matrix=cm_ga, display_labels=['Bertahan', 'Keluar'])
    disp_ga.plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title("Confusion Matrix - Fitur GA")

    plt.tight_layout()
    plot_img = plot_to_base64(plt)

    return X_train, knn_ga, selected_features, full_metrics, ga_metrics, plot_img

def compare_models_with_ga_features(df, selected_features):
    # 1. Fitur hasil GA + Cluster
    X = df[selected_features]
    y = df['Exited']

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Definisikan model
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # 4. Evaluasi model
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }

    # 5. Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    # 6. Create plot
    results_long = results_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_long.rename(columns={"index": "Model"}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_long, x="Metric", y="Score", hue="Model")
    plt.title("Perbandingan Performa Model dengan Fitur GA")
    plt.ylim(0, 1)
    plt.ylabel("Skor")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_img = plot_to_base64(plt)

    return results_df, plot_img

def compare_models_with_all_features(df):
    # 1. Semua fitur (kecuali target)
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Definisikan model
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # 4. Evaluasi model
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }

    # 5. Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    # 6. Create plot
    results_long = results_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_long.rename(columns={"index": "Model"}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_long, x="Metric", y="Score", hue="Model")
    plt.title("Perbandingan Performa Model dengan Semua Fitur")
    plt.ylim(0, 1)
    plt.ylabel("Skor")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_img = plot_to_base64(plt)

    return results_df, plot_img

def compare_ga_vs_all_features(ga_results, all_results):
    # Tambahkan kolom penanda sumber fitur
    ga_results["Fitur"] = "GA"
    all_results["Fitur"] = "Semua"

    # Gabungkan kedua DataFrame
    combined_df = pd.concat([ga_results, all_results])
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "Model"}, inplace=True)

    # Ubah ke format panjang
    combined_long = combined_df.melt(id_vars=["Model", "Fitur"], var_name="Metric", value_name="Score")

    # Buat visualisasi
    g = sns.catplot(
        data=combined_long, kind="bar",
        x="Metric", y="Score", hue="Fitur", col="Model",
        palette="Set2", height=4, aspect=0.9,
        errorbar=None
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Metric", "Skor")
    g.set(ylim=(0, 1))
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Hapus judul legend dan pindahkan legend ke bawah
    g._legend.set_title("")
    g._legend.set_bbox_to_anchor((0.5, -0.01))
    g._legend.set_loc("lower center")

    # Tambahkan judul utama dan beri ruang ekstra
    plt.subplots_adjust(top=0.85, bottom=0.2)
    plt.suptitle("Perbandingan Performa Model: Fitur GA vs Semua Fitur", fontsize=14)
    plot_img = plot_to_base64(plt)

    return plot_img


    