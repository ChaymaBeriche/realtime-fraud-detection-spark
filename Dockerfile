# Image du projet : Python 3.11 + Java 17 + PySpark
FROM python:3.11-slim-bookworm

# Java, nécessaire à Spark (default-jre-headless = OpenJDK 17 sur Debian bookworm)
# procps fournit la commande "ps", utilisée par les scripts de Spark
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Affiche les sorties Python immédiatement (sans mise en mémoire tampon)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances Python. Le code et les données ne sont pas copiés dans l'image :
# le dossier du projet est partagé avec le conteneur (voir docker-compose.yml).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
