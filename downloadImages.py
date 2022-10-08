import requests
import json 
import time
import os


# Desenha a barra de progresso de download das imagens
def draw_loader(arquivo, total, count):
	os.system('cls' if os.name == 'nt' else 'clear')
	print('Baixando imagens\n')
	print(arquivo, end='')
	print(' ' + str(count) + ' de ' + str(total))
	percent = (count/total) * 100
	print('[', end='')
	for i in range(30):
		if (i/30) * 100 >= percent:
			print('-', end='')
		else:
			print('#', end='')
	print(']{:.2f}%'.format(percent))


# Recebe o Json e devolve uma lista com os Ids e o Link para as imagens
def prepare_jason(data_request):
	data = json.loads(data_request)['data']
	link_images = []
	for i in range(len(data)):
		for j in range(len(data[i]['card_images'])):
			id = data[i]['card_images'][j]['id']
			link = data[i]['card_images'][j]['image_url']
			link_images.append([str(id), link])
	return link_images


def main():
	# Link para api para as +11000 cartas
	url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?level=4&attribute=water"
	request = requests.get(url)
	data_request = request.content

	# Prepara os links das imagens
	link_images = prepare_jason(data_request)
	quantidade_imagens = len(link_images)

	# Para cada carta busca todas as imagens e guarda no diretório dataset/
	for i in range(quantidade_imagens):
		id = link_images[i][0]
		link = link_images[i][1]
		file_name = id + '.jpg'

		# Desenha barra de progresso 
		draw_loader(file_name, quantidade_imagens, i+1)

		# Baixa o arquivo
		image = requests.get(link, allow_redirects=True)
		open('dataset/' + file_name, 'wb').write(image.content)

		# Sleep para não ultrapassar o limite permitido pela API
		time.sleep(0.05)
	print('Processo Finalizado!')


if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	main()
