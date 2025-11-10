FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Colectar archivos estáticos
RUN python manage.py collectstatic --no-input

# Puerto expuesto
EXPOSE 10000

# Comando de inicio
CMD ["gunicorn", "malware_detector.wsgi:application", "--bind", "0.0.0.0:10000"]
