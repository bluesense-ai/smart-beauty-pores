FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the ImageNet ResNet18 weights into the image so container start does not
# depend on reaching download.pytorch.org.
RUN python -c "from torch.hub import load_state_dict_from_url; load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')"

# api.py imports engines/ and core_models/ as packages and the engines load
# weights from models/ relative to the working directory, so src/ is /app.
COPY src/ .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
