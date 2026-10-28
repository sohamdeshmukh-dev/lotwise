FROM python:3.13-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENV DATABASE_URL=sqlite:////data/lotwise.db
EXPOSE 8000
CMD ["uvicorn", "lotwise.api:app", "--host", "0.0.0.0", "--port", "8000"]

