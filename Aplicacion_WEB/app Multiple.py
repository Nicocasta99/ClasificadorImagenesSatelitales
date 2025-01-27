from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import io
import cv2
import base64
import torch
import tensorflow as tf
from torchvision import transforms

app = Flask(__name__)

# Cargar los modelos
models_h5_paths = {
    'model_1_h5': 'C:/Users/nicol/Desktop/Model1_Xception.h5',
    'model_2_h5': 'C:/Users/nicol/Desktop/Model2_Xception.h5'
}
models_pth_paths = {
    'model_3_pth': 'C:/Users/nicol/Desktop/Model3.pth',
    'model_4_pth': 'C:/Users/nicol/Desktop/Model4.pth',
    'model_5_pth': 'C:/Users/nicol/Desktop/Model5.pth'
    #'model_5_pth': 'C:/Users/nicol/Desktop/Model5.pth'
}

models_h5 = {name: tf.keras.models.load_model(path) for name, path in models_h5_paths.items()}
models_pth = {name: torch.load(path) for name, path in models_pth_paths.items()}

# Definir las categorías
categorias = {
    'model_1_h5': ['Basket', 'Campo_Futbol', 'Cancha_Micro', 'Cancha_Multiple', 'Parque', 'Tenis'],
    'model_2_h5': ['baseball_field', 'basketball_court', 'football_field', 'golf_course', 'tennis_court'],
    'model_3_pth': ['baseball_diamond', 'basketball_court', 'golf_course', 'ground_track_field'],
    'model_4_pth': ['baseballdiamond', 'footballField', 'golfcourse', 'Park', 'Pond', 'tenniscourt'],
    'model_5_pth': ['baseball_diamond', 'basketball_court', 'golf_course', 'ground_track_field', 'park', 'stadium', 'tennis_court']
}

# Configuración de preprocesamiento para PyTorch y TensorFlow
transform_pth = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocesar_imagen_h5(seccion):
    seccion_redimensionada = redimensionar_imagen(seccion, (256, 256))
    seccion_normalizada = seccion_redimensionada / 255.0
    seccion_normalizada = np.expand_dims(seccion_normalizada, axis=0)
    return seccion_normalizada

def convertir_a_rgb(img):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def redimensionar_imagen(img, nuevo_tamanio):
    img_pil = Image.fromarray(img)
    img_redimensionada = img_pil.resize(nuevo_tamanio, Image.Resampling.LANCZOS)
    return np.array(img_redimensionada)

# Función para procesar la imagen
def procesar_imagen(img, grid_size, model_type):
    img = convertir_a_rgb(img)
    height, width, _ = img.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    conteo_clases = {}
    clases = categorias[model_type]
    umbral_probabilidad = 0.2 if 'h5' in model_type else 0.8

    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            seccion = img[y_start:y_end, x_start:x_end]

            if 'pth' in model_type:
                model = models_pth[model_type]
                seccion_transformada = transform_pth(seccion).unsqueeze(0)
                with torch.no_grad():
                    prediccion = model(seccion_transformada)
                    probabilidad_maxima = torch.max(prediccion).item()
                    clase_predicha = torch.argmax(prediccion).item()
            elif 'h5' in model_type:
                model = models_h5[model_type]
                seccion_preprocesada = preprocesar_imagen_h5(seccion)
                prediccion = model.predict(seccion_preprocesada)
                probabilidad_maxima = np.max(prediccion[0])
                clase_predicha = np.argmax(prediccion[0])

            nombre_clase = clases[clase_predicha]
            if probabilidad_maxima < umbral_probabilidad:
                nombre_clase = 'Nada'

            if nombre_clase in conteo_clases:
                conteo_clases[nombre_clase] += 1
            else:
                conteo_clases[nombre_clase] = 1

            texto = f"{nombre_clase}"
            posicion_texto = (x_start + 5, y_start + 20)
            cv2.putText(img, texto, posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

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
        model_type = request.form.get('model_type')
        if file and grid_size and model_type:
            try:
                grid_size = int(grid_size)
                img = Image.open(file.stream)
                img = np.array(img)

                img_procesada = procesar_imagen(img, grid_size, model_type)

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
