import numpy as np
from PIL import Image as pim
from glob import glob
import os

from semitone import to_semitone, image_to_np_array, path
from semitone import Image, np

def prompt(variants: dict):
    for number, variant in enumerate(variants.keys(), 1):
        print(f'{number} - {variant}')
    input_correct = False
    user_input = 0
    while not input_correct:
        try:
            user_input = int(input('> '))
            if user_input <= 0 or user_input > len(variants):
                raise ValueError
            input_correct = True
        except ValueError:
            print("Введите корректное значение")
    return dict(enumerate(variants.values(), 1))[user_input]


def safe_number_input(lower_bound=None, upper_bound=None):
    input_correct = False
    user_input = 0
    while not input_correct:
        try:
            user_input = int(input('> '))
            if lower_bound is not None and user_input < lower_bound:
                raise ValueError
            if upper_bound is not None and user_input > upper_bound:
                raise ValueError
            input_correct = True
        except ValueError:
            print("Введите корректное значение")
    return user_input

images = {
    "Man": 'man.png',
    "Book": 'book.png',
}

operations = {
        'Полутон': 'semitone',
        'бинаризация': 'binarisation'
}

def main():
    print('Выберите изображение:')
    selected_image = prompt(images)
    img = image_to_np_array(selected_image)

    print("Выберите обработку изображения:")
    selected_handle = prompt(operations)

    match selected_handle:
        case 'semitone':
            result = to_semitone(selected_image)
            output_name = 'output_semitone_' + selected_image 
            result.save(path.join('output', output_name))
        case _:
            exit()
    
if __name__ == "__main__":
    main()

