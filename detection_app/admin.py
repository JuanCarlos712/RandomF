from django.contrib import admin
from .models import TrainedModel

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'f1_score', 'precision', 'recall', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name']
