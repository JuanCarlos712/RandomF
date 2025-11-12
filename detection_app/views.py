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
import joblib
import random
from django.shortcuts import render
from django.conf import settings
from .models import TrainedModel

# Ruta para guardar los modelos
MODELS_DIR = os.path.join(settings.BASE_DIR, 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)

def home(request):
    context = {}
    
    # Verificar si hay modelos guardados
    saved_models = TrainedModel.objects.all()
    
    if request.method == 'POST':
        # Entrenar nuevo modelo
        context = train_new_model(request)
    else:
        # Mostrar modelo aleatorio existente
        if saved_models.exists():
            context = show_random_model(saved_models)
        else:
            # Si no hay modelos, entrenar uno nuevo
            context = train_new_model(request)
    
    # Añadir contador de modelos
    context['total_models'] = saved_models.count()
    
    return render(request, 'detection_app/results.html', context)

def train_new_model(request):
    """Entrena un nuevo modelo y lo guarda"""
    context = {}
    
    try:
        # Descargar y procesar datos (tu código existente)
        file_id = '1damckL9bh8APnI8lLbzs2f7R0Tblroq4'
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        session = requests.Session()
        response = session.get(download_url, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            download_url = f'https://drive.google.com/uc?export=download&confirm={token}&id={file_id}'
            response = session.get(download_url, stream=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            csv_path = tmp_file.name
        
        # Cargar datos
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        if df.empty:
            raise Exception("No se pudieron cargar datos del archivo CSV")
        
        # Buscar columna target
        if 'calss' not in df.columns:
            possible_targets = ['calss', 'class', 'target', 'label', 'type', 'Category']
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                df = df.rename(columns={target_col: 'calss'})
            else:
                raise Exception(f"Columna 'calss' no encontrada")
        
        df = df.dropna(subset=['calss'])
        
        # Usar muestra más pequeña para entrenamiento rápido
        if len(df) > 3000:
            df = df.sample(n=3000, random_state=42)
        
        X = df.drop('calss', axis=1)
        y = df['calss']
        
        # Convertir labels
        y_encoded, classes = pd.factorize(y)
        
        if len(classes) < 2:
            raise Exception(f"Solo se encontró una clase: {classes}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Escalar
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar Random Forest con parámetros aleatorios para variedad
        n_estimators = random.choice([50, 100, 150])
        max_depth = random.choice([10, 20, None])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
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
        plots = create_plots(model, X.columns, f1, report, cm, classes)
        
        # Top características
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:5])
        
        # Guardar modelo
        model_id = len(TrainedModel.objects.all()) + 1
        model_filename = f'model_{model_id}.joblib'
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Guardar modelo y scaler
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'classes': list(classes)
        }, model_path)
        
        # Guardar en base de datos
        trained_model = TrainedModel.objects.create(
            name=f"Modelo RF-{n_estimators}-{'NoLimit' if max_depth is None else max_depth}",
            f1_score=f1,
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            feature_importance=feature_importance,
            confusion_matrix=cm.tolist(),
            classes=list(classes),
            model_file=model_filename
        )
        
        context.update({
            'trained': True,
            'new_model': True,
            'model_name': trained_model.name,
            'f1_score': round(f1, 4),
            'precision': round(report['weighted avg']['precision'], 4),
            'recall': round(report['weighted avg']['recall'], 4),
            'plots': plots,
            'top_features': top_features,
            'classes': list(classes),
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })
        
        # Limpiar archivo temporal
        os.unlink(csv_path)
        
    except Exception as e:
        context['error'] = f"Error durante el entrenamiento: {str(e)}"
    
    return context

def show_random_model(saved_models):
    """Muestra un modelo pre-entrenado aleatorio"""
    random_model = random.choice(list(saved_models))
    
    try:
        # Cargar modelo desde archivo
        model_path = os.path.join(MODELS_DIR, random_model.model_file)
        if not os.path.exists(model_path):
            raise Exception("Archivo del modelo no encontrado")
        
        loaded_data = joblib.load(model_path)
        model = loaded_data['model']
        feature_names = loaded_data['feature_names']
        classes = loaded_data['classes']
        
        # Recrear gráficas
        feature_importance = random_model.feature_importance
        cm = np.array(random_model.confusion_matrix)
        
        plots = create_plots(model, feature_names, random_model.f1_score, 
                           {'weighted avg': {
                               'precision': random_model.precision,
                               'recall': random_model.recall
                           }}, cm, classes)
        
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:5])
        
        context = {
            'trained': True,
            'new_model': False,
            'model_name': random_model.name,
            'f1_score': round(random_model.f1_score, 4),
            'precision': round(random_model.precision, 4),
            'recall': round(random_model.recall, 4),
            'plots': plots,
            'top_features': top_features,
            'classes': classes,
            'created_at': random_model.created_at.strftime("%Y-%m-%d %H:%M"),
            'total_models': saved_models.count()
        }
        
    except Exception as e:
        context = {
            'error': f"Error cargando modelo: {str(e)}",
            'total_models': saved_models.count()
        }
    
    return context

def create_plots(model, feature_names, f1_score, report, cm, classes):
    """Crea las gráficas para mostrar"""
    plots = {}
    
    # 1. Importancia de características (Top 10)
    plt.figure(figsize=(12, 8))
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
    else:
        # Si no tiene feature_importances_, usar datos guardados
        feature_importance = {name: 0.1 for name in feature_names[:10]}
    
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
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
    
    for bar, value in zip(bars, metrics_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['metrics'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots
