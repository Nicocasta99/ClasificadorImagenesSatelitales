from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import io
import cv2
import base64
import torch
from torchvision import transforms

app = Flask(__name__)

# Cargar el modelo previamente entrenado de PyTorch
modelo_path = 'C:/Users/nicol/Desktop/Modelos/Propio/Propio_mobileNET.pth'
model = torch.load(modelo_path)
model.eval()  # Establecer el modelo en modo de evaluación

# Definir las transformaciones con normalización y redimensionamiento como en el preprocesamiento de entrenamiento
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convertir de numpy array a PIL image
    transforms.Resize((256, 256)),  # Redimensionar la imagen
    transforms.ToTensor(),  # Convertir de PIL a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización de ImageNet
])

# Función para convertir imágenes a RGB
def convertir_a_rgb(img):
    if img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 1:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Función para procesar la imagen
def procesar_imagen(img, grid_size):
    img = convertir_a_rgb(img)  # Asegúrate de que la imagen sea RGB

    height, width, _ = img.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    conteo_clases = {}
    nombres_clases = {
        0: 'Basket',
        1: 'Campo_Futbol',
        2: 'Cancha_Multiple',
        3: 'Parque',
        4: 'Tenis'
    }
    
    umbral_probabilidad = 0.5

    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            seccion = img[y_start:y_end, x_start:x_end]
            # Aplicar las transformaciones definidas
            seccion_transformada = transform(seccion)
            seccion_transformada = seccion_transformada.unsqueeze(0)  # Agregar batch dimension
            
            # Realizar predicción con el modelo de PyTorch
            with torch.no_grad():
                prediccion = model(seccion_transformada)
                probabilidad_maxima = torch.max(prediccion).item()
                clase_predicha = torch.argmax(prediccion).item()
            
            nombre_clase = nombres_clases.get(clase_predicha, 'Desconocida')
            if probabilidad_maxima < umbral_probabilidad:
                nombre_clase = 'Nada'

            if nombre_clase in conteo_clases:
                conteo_clases[nombre_clase] += 1
            else:
                conteo_clases[nombre_clase] = 1

            texto = f"{nombre_clase}"
            posicion_texto = (x_start + 5, y_start + 20)
            cv2.putText(img, texto, posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Dibujar las líneas del grid
    for row in range(1, grid_size):
        start_point = (0, row * cell_height)
        end_point = (width, row * cell_height)
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)

    for col in range(1, grid_size):
        start_point = (col * cell_width, 0)
        end_point = (col * cell_width, height)
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)

    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        grid_size = request.form.get('grid_size')
        if file and grid_size:
            try:
                grid_size = int(grid_size)
                img = Image.open(file.stream)
                img = np.array(img)

                # Procesar la imagen
                img_procesada = procesar_imagen(img, grid_size)

                # Convertir la imagen procesada a formato base64
                img_pil = Image.fromarray(img_procesada)
                buffered = io.BytesIO()
                img_pil.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                return render_template('result.html', img_data=img_str)

            except Exception as e:
                return f"Error processing the image: {e}"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
