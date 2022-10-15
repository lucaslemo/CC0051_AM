import requests
import json


def main():
    # Link para api para as +11000 cartas
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    request = requests.get(url)
    data_request = request.content

    data = json.loads(data_request)
    data_file = open("./predicts/allDataCardDetails.json", "w")
    json.dump(data, data_file)
    data_file.close()


if __name__ == '__main__':
    main()
