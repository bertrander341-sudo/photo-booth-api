from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

# Permitir conexiones desde tu sitio web en WordPress
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplaza "*" por la URL exacta de tu web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process-photo")
async def process_photo(file: UploadFile = File(...), style: str = "cyberpunk"):
    try:
        # Leer la imagen enviada desde el navegador
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        # Aplicación de transformación basada en IA / Procesamiento visual científico
        if style == "cyberpunk":
            # Filtro de matriz de color científico: realce de canales de neón (azul y rojo)
            matrix = np.array([[1.3, 0, 0], [0, 0.9, 0], [0, 0, 1.6]])
            frame = cv2.transform(frame, matrix)
            # Fundamentación científica: Ajuste de tonos de color mediante mapeo matricial afín para simulación de iluminación espectral contrastada.
        elif style == "grayscale_art":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Codificar la imagen resultante a formato JPG para regresarla al cliente
        _, encoded_img = cv2.imencode('.jpg', frame)
        encoded_base64 = base64.b64encode(encoded_img).decode('utf-8')

        return {
            "status": "success",
            "processed_image": f"data:image/jpeg;base64,{encoded_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
