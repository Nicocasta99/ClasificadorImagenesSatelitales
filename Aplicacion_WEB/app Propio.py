from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import io
import cv2
import base64  # Asegúrate de importar base64
import tensorflow as tf

app = Flask(__name__)

# Cargar el modelo previamente entrenado
modelo_path = 'C:/Users/nicol/Desktop/PropioCopy_Xception.h5'
model = tf.keras.models.load_model(modelo_path)

# Funciones de preprocesamiento
def redimensionar_imagen(img, nuevo_tamanio):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_redimensionada = img_pil.resize(nuevo_tamanio, Image.Resampling.LANCZOS)
    return np.array(img_redimensionada)

def verificar_tamanio(img, tamanio_referencia):
    return img.shape[:2] == tamanio_referencia

def normalizar_imagen(img):
    img_normalizada = img / 255.0
    media = np.mean(img_normalizada, axis=(0, 1))
    img_normalizada -= media
    return img_normalizada

def convertir_a_rgb(img):
    # Verifica si la imagen tiene 4 canales (BGRA)
    if img.shape[2] == 4:  # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Primero quita la transparencia y convierte a BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Luego convierte de BGR a RGB
    elif img.shape[2] == 1:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def procesar_imagen(img, grid_size):
    img = convertir_a_rgb(img)  # Asegúrate de que la imagen sea RGB

    height, width, _ = img.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    conteo_clases = {}
    nombres_clases = {
        0: 'Campo de baloncesto',
        1: 'Campo de futbol',
        2: 'Cancha multiple',
        3: 'Parque',
        4: 'Cancha de tenis'
    }

    umbral_probabilidad = 0.2

    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width

            seccion = img[y_start:y_end, x_start:x_end]
            seccion_redimensionada = redimensionar_imagen(seccion, (256, 256))
            seccion_normalizada = normalizar_imagen(seccion_redimensionada)
            seccion_normalizada = np.expand_dims(seccion_normalizada, axis=0)
            prediccion = model.predict(seccion_normalizada)
            probabilidad_maxima = np.max(prediccion[0])
            clase_predicha = np.argmax(prediccion[0])
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
