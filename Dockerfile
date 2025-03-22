FROM python:3.10.16

# Create a non-root user to run the application
RUN useradd -m appuser

# Create directories for data with proper permissions
RUN mkdir -p /app/data/nltk_data && chown -R appuser:appuser /app

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY . .

# Set proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0"]