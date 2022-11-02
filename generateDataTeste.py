import os
import torchvision.transforms as transforms
from PIL import Image


def get_positive(image):
    change_img = transforms.Compose(
        [transforms.ColorJitter(brightness=(0.8,1.5), contrast=(0.5, 1.5), saturation=(0.8,1.3))]
    )
    return change_img(image)


def load_data_paths(path):
    images_path = []
    for card in os.listdir(path):
        item = {
            'path': os.path.join(path, card),
            'card_id': int(card.strip('.jpg'))
        }
        images_path.append(item)
    return images_path


def main():
    dir_path = os.getcwd()
    data_training_path = os.path.join(dir_path, 'train_test_dataset')
    images_path = load_data_paths(data_training_path)
    for image_path in images_path:
        image = Image.open(image_path['path'])
        new_image = get_positive(image)
        file_name = str(image_path['card_id']) + '.jpg'
        destination = os.path.join(dir_path, 'test_dataset\\' + file_name)
        new_image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	main()