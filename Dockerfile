# Use an official lightweight Python image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optionally, specify a default command (e.g., to run a specific script)
# CMD ["python", "fetch_qqq_data.py"]
