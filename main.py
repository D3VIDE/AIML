from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_session import Session
import os
import uuid
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import tempfile
import atexit
from deap import creator, base
import warnings

# Suppress DEAP warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize DEAP types safely
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

from churn_ai import (
    preprocess_data,
    load_data,
    visualisasi_churn,
    clustering,
    final_clustering,
    analysis_cluster,
    dataframe_to_html,
    genetic_algorithm,
    compare_ga_vs_all_features,
    compare_models_with_ga_features,
    knn_comparison,
    confusion_matrix_comparison,
    compare_models_with_all_features
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp_session_data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Configure server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_NAME'] = 'churn_analysis_session'
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    TEMP_FOLDER=TEMP_FOLDER
)

# Initialize server-side session
Session(app)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_temp_data(session_id):
    """Clean up temporary files for specific session"""
    try:
        if session_id:
            temp_file = os.path.join(app.config['TEMP_FOLDER'], f'{session_id}.pkl')
            if os.path.exists(temp_file):
                os.remove(temp_file)
                app.logger.info(f"Cleaned up temp file: {temp_file}")
    except Exception as e:
        app.logger.error(f"Failed to cleanup temp data: {str(e)}")

def save_temp_data(session_id, data):
    """Save large data to temporary file"""
    try:
        temp_file = os.path.join(app.config['TEMP_FOLDER'], f'{session_id}.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f)
        return temp_file
    except Exception as e:
        app.logger.error(f"Failed to save temp data: {str(e)}")
        raise

def load_temp_data(file_path):
    """Load data from temporary file with error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Temp file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        app.logger.error(f"Failed to load temp data: {str(e)}")
        raise

def cleanup_all_temp_files():
    """Clean up all temporary files on shutdown"""
    try:
        for filename in os.listdir(app.config['TEMP_FOLDER']):
            file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Failed to cleanup all temp files: {str(e)}")

# Register cleanup on exit
atexit.register(cleanup_all_temp_files)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate unique session ID if not exists
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            
            session['data_file'] = filepath
            
            if 'analyse' in request.form:
                return redirect(url_for('analyse'))
            
        else:
            flash('Invalid file type. Please upload CSV or Excel files.', 'error')
    
    return render_template('index.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if 'data_file' not in session:
        flash('No data uploaded', 'error')
        return redirect(url_for('index'))
    
    filepath = session['data_file']
    
    try:
        current_step = int(request.form.get('current_step', 1))
        
        # Step 1: Load and show basic data
        if current_step == 1:
            df_raw = load_data(filepath)
            df_processed, df_unscaled = preprocess_data(df_raw)
            churn_plots = visualisasi_churn(df_processed)
            X, elbow_silhouette = clustering(df_processed)
            
            # Save data to temp file instead of session
            data_to_store = {
                'df_processed': df_processed,
                'df_unscaled': df_unscaled,
                'X': X
            }
            save_temp_data(session['session_id'], data_to_store)
            
            raw_data_html = dataframe_to_html(df_raw)
            processed_data_html = dataframe_to_html(df_processed)
            
            return render_template('analyse.html',
                                step=1,
                                raw_data_html=raw_data_html,
                                processed_data_html=processed_data_html,
                                churn_plots=churn_plots,
                                elbow_silhouette=elbow_silhouette)
        
        # Load data from temp file for subsequent steps
        data = load_temp_data(os.path.join(app.config['TEMP_FOLDER'], f"{session['session_id']}.pkl"))
        
        # Step 2: Get K value and show clustering
        if current_step == 2:
            k_value = int(request.form.get('k_value', 3))
            df_clustered, cluster_plot = final_clustering(data['df_processed'], data['X'], final_k=k_value)
            
            # Update temp file with clustered data
            data['df_clustered'] = df_clustered
            save_temp_data(session['session_id'], data)
            
            return render_template('analyse.html',
                                step=2,
                                cluster_plot=cluster_plot,
                                k_value=k_value)
        
        # Step 3: Show cluster analysis and get GA params
        elif current_step == 3:
            df_unscaled, cluster_analysis_plots = analysis_cluster(data['df_clustered'], data['df_unscaled'])
            
            # Update temp file
            data['df_unscaled'] = df_unscaled
            save_temp_data(session['session_id'], data)
            
            return render_template('analyse.html',
                                step=3,
                                cluster_analysis_plots=cluster_analysis_plots)
        
        # Step 4: Run GA and show final results
        elif current_step == 4:
            pop_size = int(request.form.get('pop_size', 30))
            ngen = int(request.form.get('ngen', 20))
            
            selected_features = genetic_algorithm(data['df_clustered'], pop_size=pop_size, ngen=ngen)
            
            # Process GA results
            knn_ga, knn_plot, knn_results = knn_comparison(data['df_clustered'], selected_features)
            X_train, knn_model, selected_features, full_metrics, ga_metrics, confusion_plot = confusion_matrix_comparison(data['df_clustered'], selected_features)
            ga_results, ga_model_plot = compare_models_with_ga_features(data['df_clustered'], selected_features)
            all_results, all_model_plot = compare_models_with_all_features(data['df_clustered'])
            comparison_plot = compare_ga_vs_all_features(ga_results, all_results)
            
            # Convert to HTML
            knn_results_html = dataframe_to_html(pd.DataFrame(knn_results.items(), columns=['Model', 'Accuracy']))
            full_metrics_html = dataframe_to_html(pd.DataFrame(full_metrics.items(), columns=['Metric', 'Score']))
            ga_metrics_html = dataframe_to_html(pd.DataFrame(ga_metrics.items(), columns=['Metric', 'Score']))
            ga_results_html = dataframe_to_html(ga_results)
            all_results_html = dataframe_to_html(all_results)

            # Save model for prediction
            model_data = {
                'knn_ga': knn_model,
                'X_train': X_train,
                'df_unscaled': data['df_unscaled'],
                'selected_features': selected_features
            }
            temp_file = save_temp_data(session['session_id'], model_data)
            
            session.update({
                'analysis_complete': True,
                'temp_file': temp_file,
                'selected_features': selected_features
            })
            
            return render_template('analyse.html',
                                step=4,
                                selected_features=selected_features,
                                knn_plot=knn_plot,
                                knn_results_html=knn_results_html,
                                confusion_plot=confusion_plot,
                                full_metrics_html=full_metrics_html,
                                ga_metrics_html=ga_metrics_html,
                                ga_results_html=ga_results_html,
                                all_results_html=all_results_html,
                                ga_model_plot=ga_model_plot,
                                all_model_plot=all_model_plot,
                                comparison_plot=comparison_plot,
                                show_prediction_button=True)
        
    except Exception as e:
        flash(f'Analysis error at step {current_step}: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/predict_manual', methods=['GET', 'POST'])
def predict_manual():
    if 'data_file' not in session:
        flash('No data uploaded', 'error')
        return redirect(url_for('index'))
    
    if not session.get('analysis_complete'):
        flash('Please complete analysis first', 'error')
        return redirect(url_for('analyse'))
    
    if 'temp_file' not in session or not os.path.exists(session['temp_file']):
        flash('Analysis data expired. Please re-run analysis.', 'error')
        return redirect(url_for('analyse'))
    
    try:
        model_data = load_temp_data(session['temp_file'])
        knn_ga = model_data['knn_ga']
        X_train = model_data['X_train']
        df_unscaled = model_data['df_unscaled']
        selected_features = model_data['selected_features']
        
        if request.method == 'POST':
            geo_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
            gender_mapping = {'Male': 1, 'Female': 0}
            
            input_data = {}
            input_values = {}
            
            for feature in selected_features:
                value = request.form.get(feature, '').strip()
                
                if feature == 'Geography':
                    geo_input = value.title() if value else 'France'
                    input_data[feature] = geo_mapping.get(geo_input, 0)
                    input_values[feature] = geo_input
                elif feature == 'Gender':
                    gender_input = value.title() if value else 'Female'
                    input_data[feature] = gender_mapping.get(gender_input, 0)
                    input_values[feature] = gender_input
                elif feature == 'Tenure':
                    try:
                        tenure_value = int(float(value))
                        if tenure_value < 0 or tenure_value > 10:
                            flash('Tenure must be between 0 and 10 years', 'error')
                            return redirect(url_for('predict_manual'))
                        input_data[feature] = tenure_value
                        input_values[feature] = tenure_value
                    except ValueError:
                        flash('Please enter a valid number for Tenure (0-10)', 'error')
                        return redirect(url_for('predict_manual'))
                else:
                    try:
                        if feature in ['HasCrCard', 'IsActiveMember']:
                            input_data[feature] = int(float(value)) if value else 0
                        else:
                            input_data[feature] = float(value) if value else 0.0
                        input_values[feature] = value
                    except ValueError:
                        flash(f'Invalid input for {feature}. Please enter a valid number.', 'error')
                        return redirect(url_for('predict_manual'))
            
            # Create and scale input DataFrame
            input_df_raw = pd.DataFrame([input_data])
            numeric_cols = [col for col in selected_features 
                          if col in ['CreditScore', 'Age', 'Tenure', 'Balance', 
                                    'NumOfProducts', 'EstimatedSalary']]
            
            input_df_scaled = input_df_raw.copy()
            for col in numeric_cols:
                mean = df_unscaled[col].iloc[X_train.index].mean()
                std = df_unscaled[col].iloc[X_train.index].std()
                input_df_scaled[col] = (input_df_scaled[col] - mean) / std if std != 0 else 0
            
            input_df_scaled = input_df_scaled[selected_features]
            
            # Make prediction
            pred = knn_ga.predict(input_df_scaled)[0]
            proba = knn_ga.predict_proba(input_df_scaled)[0][1]
            
            session['prediction_result'] = {
                'prediction': '❌ CHURN' if pred == 1 else '✅ BERTAHAN',
                'probability': f"{proba:.2%}",
                'input_data': input_values,
                'features': selected_features
            }
            
            return redirect(url_for('prediction_result'))
        
        return render_template('predict_manual.html', 
                            features=selected_features,
                            geo_options=['France', 'Germany', 'Spain'],
                            gender_options=['Male', 'Female'])
    
    except FileNotFoundError:
        flash('Analysis data expired. Please re-run analysis.', 'error')
        return redirect(url_for('analyse'))
    except Exception as e:
        flash(f'System error: {str(e)}', 'error')
        app.logger.error(f"Prediction error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/prediction_result')
def prediction_result():
    if 'prediction_result' not in session:
        flash('No prediction results available', 'error')
        return redirect(url_for('index'))
    
    result = session['prediction_result']
    return render_template('prediction_result.html', 
                         result=result,
                         numeric_features=['CreditScore', 'Age', 'Tenure', 'Balance', 
                                         'NumOfProducts', 'EstimatedSalary'])



if __name__ == '__main__':
    app.run(debug=True)