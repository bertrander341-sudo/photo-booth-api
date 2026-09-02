from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process-photo")
async def process_photo(
    file: UploadFile = File(...), 
    style: str = Query("original_enhanced"),
    border_color: str = Query("none"),
    resolution: str = Query("1080p"),
    print_size: str = Query("none"),
    skin_smooth: bool = Query(False)
):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        # 1. Estilos visuales
        if style == "cyberpunk":
            matrix = np.array([[1.3, 0, 0], [0, 0.9, 0], [0, 0, 1.6]])
            frame = cv2.transform(frame, matrix)
        elif style == "grayscale_art":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif style == "warm_portrait":
            matrix = np.array([[1.1, 0, 0], [0, 1.05, 0], [0, 0, 0.95]])
            frame = cv2.transform(frame, matrix)
        elif style == "cinematic_chiaroscuro":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            matrix = np.array([[0.9, 0, 0], [0, 1.0, 0], [0, 0, 1.2]])
            frame = cv2.transform(frame, matrix)

        # 2. Retoque suave de piel
        if skin_smooth:
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Tamaños de impresión (Carnet, Pasaporte, Jumbo)
        h, w = frame.shape[:2]
        if print_size in ["carnet", "pasaporte", "jumbo"]:
            ratios = {"carnet": 3.0/4.0, "pasaporte": 4.0/5.0, "jumbo": 2.0/3.0}
            target_ratio = ratios[print_size]
            current_ratio = w / h
            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                start_x = (w - new_w) // 2
                frame = frame[:, start_x:start_x + new_w]
            else:
                new_h = int(w / target_ratio)
                start_y = (h - new_h) // 2
                frame = frame[start_y:start_y + new_h, :]

        # 4. Resoluciones (1080p, 4K, 8K) con nitidez adaptativa
        height, width = frame.shape[:2]
        target_width = 7680 if resolution == "8k" else (3840 if resolution == "4k" else 1920)
        target_height = int(height * (target_width / width))
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        strength = 1.8 if resolution in ["4k", "8k"] else 1.3
        frame = cv2.addWeighted(frame, strength, gaussian, -(strength - 1.0), 0)

        # 5. Marcos personalizados
        if border_color != "none":
            colors = {
                "white": (255, 255, 255), 
                "blue": (255, 128, 0), 
                "black": (0, 0, 0), 
                "pink": (203, 192, 255)
            }
            bgr_color = colors.get(border_color, (255, 255, 255))
            border_thickness = int(max(frame.shape[0], frame.shape[1]) * 0.025)
            frame = cv2.copyMakeBorder(frame, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=bgr_color)

        # 6. Codificación y respuesta en base64
        _, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        encoded_base64 = base64.b64encode(encoded_img).decode('utf-8')

        return {
            "status": "success", 
            "processed_image": f"data:image/jpeg;base64,{encoded_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
