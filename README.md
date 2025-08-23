# ECG Streamlit App



### 1. Local Development

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

### 2. Running with Docker

1. **Build the Docker image:**
    ```bash
    docker build -t streamlit-app .
    ```

2. **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 streamlit-app
    ```

3. **Access the app:**  
    Open your browser and go to [http://localhost:8501](http://localhost:8501)

### 3. Example `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```