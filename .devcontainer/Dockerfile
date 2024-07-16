# Base Image: Start with the official Python 3.11 slim image. 
# It's a minimal image to keep things small and efficient.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Working Directory: Set the main working directory inside the container
WORKDIR /app

ENV PYTHONPATH="/app/:${PYTHONPATH}" 

# Copy requirements file: Allows you to install your Python dependencies
COPY requirements.txt .

# Install Dependencies:  Use pip to install everything listed in your requirements.txt
RUN pip install -r requirements.txt

# Copy Your Application Code: Add your actual Python code to the container
COPY . .

# Update package lists and install the required packages in a single RUN command
RUN apt-get update && \
    apt-get install -y ffmpeg procps graphviz && \
    # Create the sudo script
    printf '#!/bin/bash\n' > /usr/bin/sudo && \
    printf 'echo "sudo was called to run: $@"\n' >> /usr/bin/sudo && \
    printf '"$@"\n' >> /usr/bin/sudo && \
    chmod +x /usr/bin/sudo

# Optional: Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Default Command (Optional): 
# Uncomment if you want a specific command to run when the container starts
# For example, if your main script is called "main.py":
# CMD ["python", "main.py"]
