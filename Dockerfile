# 1. Start from a base image
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
# 3. Copy dependency file first (for caching - see tip below)
COPY requirements.prod.txt ./
# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.prod.txt
# 5. Copy your application code and model files
COPY . .
# Ensure mlruns directory exists (if needed)
RUN mkdir -p mlruns/1/models
# 6. Tell Docker which port your app uses - since we are using uvicorn with --port 8000, we expose that port
EXPOSE 8000
# 7. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]