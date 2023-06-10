import os
import json
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import save_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

diretorio_imagens = "images"
arquivo_modelo = "modelo.h5"
altura_normalizada = 224
largura_normalizada = 224
rotulos_config_file = 'rotulos_config.json'

# Verifica se um arquivo é uma imagem válida
def is_imagem(arquivo):
    try:
        Image.open(arquivo)
        return True
    except:
        return False

# Percorre o diretório de imagens e adiciona os caminhos das imagens à lista
def percorrer_diretorio(diretorio):
    imagens = []
    for nome_arquivo in os.listdir(diretorio):
        caminho = os.path.join(diretorio, nome_arquivo)
        if os.path.isdir(caminho):
            # Se for um diretório, chama a função recursivamente
            imagens += percorrer_diretorio(caminho)
        else:
            # Verifica se o arquivo é uma imagem
            if is_imagem(caminho):
                imagens.append(caminho)
    return imagens

# Extrai o rótulo de uma imagem a partir do nome do arquivo
def extrair_rotulo(imagem_path):
    nome_arquivo = os.path.splitext(os.path.basename(imagem_path))[0]
    substring = nome_arquivo.split("_")[0]
    return substring

# Normaliza as imagens redimensionando-as para uma altura e largura específicas
def normalizar_imagens(lista_imagens, nova_altura, nova_largura):
    imagens_normalizadas = []
    for i, imagem_path in enumerate(lista_imagens):
        print(f"Normalizando imagem {i+1} de {len(lista_imagens)}: {imagem_path}")
        imagem = Image.open(imagem_path)
        imagem = imagem.resize((nova_largura, nova_altura))       
        if imagem.size == (nova_largura, nova_altura) and imagem.mode == "RGB":
            imagem_array = np.array(imagem)
            imagem_normalizada = imagem_array / 255.0
            imagens_normalizadas.append(imagem_normalizada)
        else:
            print(f"Ignorando imagem {i+1} devido a dimensões ou número de canais incorretos.")        
    
    return imagens_normalizadas

# Codifica os rótulos em valores inteiros usando o LabelEncoder
def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded, label_encoder

# Salva os rótulos únicos em um arquivo de configuração
def salvar_rotulos_config(rotulos, encode, nome_arquivo):
    rotulos_encoded = {rotulo: int(codigo) for rotulo, codigo in zip(rotulos, encode)}
    with open(nome_arquivo, 'w') as file:
        json.dump(rotulos_encoded, file)

print("Iniciando treinamento...")

# Carrega as imagens do diretório
imagens = percorrer_diretorio(diretorio_imagens)

random.shuffle(imagens)

imagens = imagens[:3000]

# Extrai os rótulos das imagens
rotulos = [extrair_rotulo(imagem) for imagem in imagens]

print("Total de imagens:", len(imagens))


print("Normalizando imagens...")
# Normaliza as imagens
imagens_normalizadas = normalizar_imagens(imagens, altura_normalizada, largura_normalizada)

print("Codificando rótulos...")
rotulos_encoded, _ = encode_labels(rotulos)
salvar_rotulos_config(rotulos, rotulos_encoded, rotulos_config_file)

print("Separando imagens de treinamento e teste...")
# Divisão dos dados em treinamento e teste
split = int(0.8 * len(imagens))  # 80% para treinamento, 20% para teste
imagens_treino, imagens_teste = imagens_normalizadas[:split], imagens_normalizadas[split:]
rotulos_treino_encoded, rotulos_teste_encoded = rotulos_encoded[:split], rotulos_encoded[split:]

print("Imagens de treinamento:", len(imagens_treino))
print("Imagens de teste:", len(imagens_teste))

# Conversão para arrays numpy
print("Convertendo para arrays numpy...")
imagens_treino_array = np.array(imagens_treino)
imagens_teste_array = np.array(imagens_teste)

# Expande as dimensões das imagens para compatibilidade com o modelo
imagens_treino_array = np.expand_dims(imagens_treino, axis=-1)
imagens_teste_array = np.expand_dims(imagens_teste, axis=-1)

# Criação do modelo
print("Criando modelo...")
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(altura_normalizada, largura_normalizada, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(rotulos), activation='softmax')
])

# Compilação do modelo
print("Compilando modelo...")
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
print("Iniciando treinamento do modelo...")
modelo.fit(imagens_treino_array, rotulos_treino_encoded, epochs=10, batch_size=256, validation_data=(imagens_teste_array, rotulos_teste_encoded))

# Salvando o modelo
print("Salvando o modelo...")
save_model(modelo, arquivo_modelo)

print("Treinamento concluído e modelo salvo.")
