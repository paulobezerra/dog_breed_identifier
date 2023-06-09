import imghdr
from io import BytesIO
import json
import os
import uuid
import requests
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn.calibration import LabelEncoder
from tensorflow import keras


altura_normalizada = 224
largura_normalizada = 224
app = Flask(__name__)
model = keras.models.load_model('modelo.h5')
diretorio = "images"
imagens = []
rotulos_config_file = 'rotulos_config.json'

racas = []
encodes = []

def is_imagem(arquivo):
    try:
        Image.open(arquivo)
        return True
    except:
        return False
    

def encode_labels(labels):
    # Crie uma instância do LabelEncoder
    label_encoder = LabelEncoder()

    # Ajuste o encoder aos rótulos fornecidos
    label_encoder.fit(labels)

    # Transforme os rótulos em valores inteiros
    labels_encoded = label_encoder.transform(labels)

    return labels_encoded, label_encoder

def decode_labels(encoded_labels, label_encoder):
    # Converta rótulos inteiros de volta aos rótulos originais
    labels_decoded = label_encoder.inverse_transform(encoded_labels)

    return labels_decoded


def preprocess_image(image, nova_altura, nova_largura):
    imagem = Image.open(image)
    imagem_normalizada = imagem.resize((nova_largura, nova_altura))
    salvar_imagem(imagem_normalizada, "logs")
    imagem_array = np.array(imagem_normalizada)
    return np.expand_dims(imagem_array, axis=0)

def salvar_imagem(imagem, diretorio):
    nome_aleatorio = str(uuid.uuid4()) + ".jpg"  # Gerar um nome aleatório com a mesma extensão
    caminho_salvar = os.path.join(diretorio, nome_aleatorio)
    print(caminho_salvar)
    imagem.save(caminho_salvar)

def predict_breed(image_url):
    # Baixar a imagem da URL fornecida
    response = requests.get(image_url)
    image = BytesIO(response.content)

    # Pré-processar a imagem
    preprocessed_image = preprocess_image(image, altura_normalizada, largura_normalizada)

    # Fazer a previsão usando o modelo carregado
    prediction = model.predict(preprocessed_image)
    # Obter o índice da classe com maior probabilidade
    predicted_index = np.argmax(prediction)
    # Mapear o índice para o nome da raça
    index = encodes.index(predicted_index)  # Encontra o índice de X no array encodes
    raca = racas[index]
    return raca

def carregar_rotulos_config(nome_arquivo):
    with open(nome_arquivo, 'r') as file:
        rotulos_encoded = json.load(file)
    
    rotulos = list(rotulos_encoded.keys())
    encode = list(rotulos_encoded.values())
    
    return rotulos, encode

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_url' not in request.json:
        return jsonify({'error': 'Image URL not provided'}), 400

    image_url = request.json['image_url']

    partes = image_url.split('.')
    extensao = partes[-1].lower()
    
    if extensao != 'jpeg' and extensao != 'jpg':
        return jsonify({'error': 'Image URL is not a JPEG'}), 400

    breed = predict_breed(image_url)

    return jsonify({'breed': breed}), 200


if __name__ == '__main__':

    racas, encodes = carregar_rotulos_config(rotulos_config_file)
    
    app.run()
