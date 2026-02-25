FROM python:3.11-slim

WORKDIR /app

# Copy dependency files
COPY requirements.txt . 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run training script as a module
CMD ["python", "-m", "src.train"]