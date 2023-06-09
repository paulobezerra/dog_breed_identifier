import requests
import os
import shutil
import urllib.parse

endpoint_racas = "https://dog.ceo/api/breeds/list/all"
endpoint_fotos = "https://dog.ceo/api/breed/#breed#/images" 
diretorio_destino = "images" 

shutil.rmtree(diretorio_destino)

response_racas = requests.get(endpoint_racas)
data = response_racas.json()  

racas = data["message"].keys()

for nome_raca in racas:
    print (f"Baixando fotos da raça {nome_raca}...")

    response_fotos = requests.get(endpoint_fotos.replace("#breed#", nome_raca))
    fotos = response_fotos.json() 

    if response_fotos.status_code != 200: 
        print(f"Erro ao obter fotos da raça {nome_raca}.")
        continue

    diretorio_raca = os.path.join(diretorio_destino, nome_raca)
    os.makedirs(diretorio_raca, exist_ok=True)

    for i, foto_url in enumerate(fotos["message"]):
        response_imagem = requests.get(foto_url)

        if response_imagem.status_code != 200:
            print(f"Erro ao obter foto {i+1} da raça {nome_raca}.")
            continue
        
        path = urllib.parse.urlparse(foto_url).path
        parts = path.split("/")
        nome_raca_url = parts[2]
        nome_arquivo = f"{nome_raca_url}_{i+1}.jpg"
        caminho_arquivo = os.path.join(diretorio_raca, nome_arquivo)

        with open(caminho_arquivo, "wb") as arquivo:
            arquivo.write(response_imagem.content)

        print(f"Foto {i+1} da raça {nome_raca} baixada e salva.")

print("Download das fotos concluído!")
