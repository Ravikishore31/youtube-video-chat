FROM python:3.12-slim

WORKDIR /app
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .

ENV PORT=8501 REQUIRE_USER_KEYS=true ANONYMIZED_TELEMETRY=false
EXPOSE 8501
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port \"${PORT:-8501}\" --server.headless true --browser.gatherUsageStats false"]
