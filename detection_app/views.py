import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import io
import base64
import os
import requests
import tempfile
from django.shortcuts import render
from django.conf import settings

def home(request):
    context = {}
    
    if request.method == 'POST' or 'train' in request.GET:
        try:
            # URL directa de descarga de Google Drive
            file_id = '1damckL9bh8APnI8lLbzs2f7R0Tblroq4'
            download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            
            # Descargar dataset
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Manejar archivos grandes de Google Drive
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            if token:
                download_url = f'https://drive.google.com/uc?export=download&confirm={token}&id={file_id}'
                response = session.get(download_url, stream=True)
            
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                csv_path = tmp_file.name
            
            # Verificar que el archivo no esté vacío
            file_size = os.path.getsize(csv_path)
            if file_size == 0:
                raise Exception("El archivo descargado está vacío")
            
            # Cargar y procesar datos
            df = pd.read_csv(csv_path)
            
            # Verificar que se cargaron datos
            if df.empty:
                raise Exception("No se pudieron cargar datos del archivo CSV")
            
            # Verificar que existe la columna 'calss'
            if 'calss' not in df.columns:
                raise Exception(f"Columna 'calss' no encontrada. Columnas disponibles: {list(df.columns)}")
            
            X = df.drop('calss', axis=1)
            y = df['calss'].factorize()[0]
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Escalar
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar Random Forest
            model = RandomForestClassifier(
                n_estimators=50,  # Reducido para testing
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Predecir y calcular métricas
            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Crear gráficas
            plots = create_plots(model, X.columns, f1, report, cm)
            
            # Top características
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            top_features = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:5])
            
            context.update({
                'trained': True,
                'f1_score': round(f1, 4),
                'precision': round(report['weighted avg']['precision'], 4),
                'recall': round(report['weighted avg']['recall'], 4),
                'plots': plots,
                'top_features': top_features
            })
            
            # Limpiar archivo temporal
            os.unlink(csv_path)
            
        except Exception as e:
            context['error'] = f"Error durante el entrenamiento: {str(e)}"
            # Debug: imprimir información del error
            print(f"DEBUG ERROR: {str(e)}")
    
    return render(request, 'detection_app/results.html', context)

def create_plots(model, feature_names, f1_score, report, cm):
    """Crea las gráficas para mostrar"""
    plots = {}
    
    # 1. Importancia de características (Top 10)
    plt.figure(figsize=(12, 8))
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    top_features = dict(sorted(feature_importance.items(), 
                             key=lambda x: x[1], reverse=True)[:10])
    
    plt.barh(list(top_features.keys()), list(top_features.values()))
    plt.title('Top 10 Características Más Importantes - Random Forest')
    plt.xlabel('Importancia')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['feature_importance'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 2. Matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Random Forest')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['confusion_matrix'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 3. Métricas del modelo
    plt.figure(figsize=(8, 6))
    metrics_data = {
        'F1-Score': f1_score,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall']
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=colors)
    plt.title('Métricas del Modelo Random Forest')
    plt.ylabel('Puntuación')
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, metrics_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['metrics'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots
