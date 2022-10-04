import requests
import json 
import time
import os


def draw_loader():
	os.system('cls' if os.name == 'nt' else 'clear')
	print('[--------------------]{}').format(1)


def main():
	# Link para api para as +11000 cartas
	url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
	request = requests.get(url)
	print('json recebido')

	# Transforma o json em um dict
	dicionario_cartas = json.loads(request.content)
	quantidade_cartas = len(dicionario_cartas['data'])

	# Para cada carta busca todas as imagens e guarda no diretório dataset/
	for i in range(quantidade_cartas):
		quantidade_imagens = len(dicionario_cartas['data'][i]['card_images'])
		for j in range(quantidade_imagens):
			link = dicionario_cartas['data'][i]['card_images'][j]['image_url']
			id = dicionario_cartas['data'][i]['card_images'][j]['id']
			print('baixando imagem ' + str(id) + '.jpg')
			image = requests.get(link, allow_redirects=True)
			open('dataset/' + str(id) + '.jpg', 'wb').write(image.content)
			time.sleep(0.05)


if __name__ == '__main__':
	main()
