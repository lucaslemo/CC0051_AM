import requests
import json 


def main():
	url = "https://db.ygoprodeck.com/api/v7/cardinfo.php?archetype=Blue-Eyes"
	request = requests.get(url)
	print('json recebido')
	dict = json.loads(request.content)
	for i in range(len(dict['data'])):
		for j in range(len(dict['data'][i]['card_images'])):
			link = dict['data'][i]['card_images'][j]['image_url']
			id = dict['data'][i]['card_images'][j]['id']
			print('baixando imagem '+str(id)+'.jpg')
			image = requests.get(link, allow_redirects=True)
			open('dataset/'+str(id)+'.jpg', 'wb').write(image.content)


if __name__ == '__main__':
	main()
