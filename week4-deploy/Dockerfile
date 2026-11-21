FROM python:3.12-slim
WORKDIR /srv
COPY week1-generation-pipeline /srv/week1-generation-pipeline
COPY week4-safety /srv/week4-safety
COPY week4-deploy /srv/week4-deploy
WORKDIR /srv/week4-deploy
RUN pip install --no-cache-dir fastapi uvicorn pydantic
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
