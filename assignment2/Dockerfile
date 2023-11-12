FROM python:3.9-slim

# Create and set the working directory
WORKDIR /app

# Install the required dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

CMD ["python", "assignment2.py"]