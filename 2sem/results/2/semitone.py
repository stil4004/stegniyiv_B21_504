from PIL import Image
import numpy as np
from os import path


def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(path.join('input', image_name)).convert('RGB')
    return np.array(img_src)


# Fhotoshop semitone.
def semitone(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 *
            img[:, :, 2]).astype(np.uint8)


def to_semitone(img_name):
    img = image_to_np_array(img_name)
    return Image.fromarray(semitone(img), 'L')


if __name__ == '__main__':
    result = to_semitone('house.png')
    print('Введите название сохраненного изображения (оставьте пустым, чтобы \
не сохранять)')
    selected_path = input()
    if selected_path:
        result.save(path.join('pictures_results', selected_path))
