from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Penting untuk generate plot di background
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from churn_ai import (
    preprocess_data,
    load_data,
    visualisasi_churn,
    clustering,
    final_clustering,
    analysis_cluster,
    dataframe_to_html
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                session['data_file'] = filepath
                return redirect(url_for('analyse_data'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Format file tidak diizinkan. Harap upload file CSV atau Excel', 'error')
            return redirect(request.url)
            
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse_data():
    if 'data_file' not in session:
        flash('Tidak ada data yang diupload', 'error')
        return redirect(url_for('index'))
    
    filepath = session['data_file']
    
    try:
        # 1. Load data
        df_raw = load_data(filepath)
        
        # 2. Preprocess data
        df_processed, df_unscaled = preprocess_data(df_raw)
        
        # 3. Visualisasi Churn
        churn_plots = visualisasi_churn(df_processed)
        
        # 4. Clustering
        X, elbow_silhouette = clustering(df_processed)
        
        # 5. Final Clustering
        df_clustered, cluster_plot = final_clustering(df_processed, X)
        
        # 6. Analisis Cluster
        df_unscaled, cluster_analysis_plots = analysis_cluster(df_clustered, df_unscaled)
        
        # Convert dataframes to HTML
        raw_data_html = dataframe_to_html(df_raw)
        processed_data_html = dataframe_to_html(df_processed)
        
        
        # Render template dengan semua hasil analisis
        return render_template('analyse.html',
                             raw_data_html=raw_data_html,
                             processed_data_html=processed_data_html,
                             churn_plots=churn_plots,
                             elbow_silhouette=elbow_silhouette,
                             cluster_plot=cluster_plot,
                             cluster_analysis_plots=cluster_analysis_plots)
    
    except Exception as e:
        flash(f'Error dalam menganalisis data: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)