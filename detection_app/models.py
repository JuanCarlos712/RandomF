from django.db import models
import json

class TrainedModel(models.Model):
    name = models.CharField(max_length=200)
    f1_score = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    feature_importance = models.JSONField()  # Almacena importancia de características
    confusion_matrix = models.JSONField()   # Almacena matriz de confusión
    classes = models.JSONField()            # Almacena las clases
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.CharField(max_length=500)  # Ruta del archivo del modelo
    
    def __str__(self):
        return f"{self.name} (F1: {self.f1_score:.4f})"
