import argparse
import pickle

from datasets.sort_of_clevr.color import Color
from datasets.sort_of_clevr.shape import Circle
from datasets.sort_of_clevr.shape import Rectangle
from datasets.sort_of_clevr.sort_of_clevr import SortOfCLEVR


_background_color = Color('light_gray', (230, 230, 230))


_colors = [
    Color('red', (252, 54, 59)),
    Color('green', (81, 178, 82)),
    Color('blue', (103, 107, 251)),
    Color('yellow', (255, 253, 93)),
    Color('orange', (253, 115, 34)),
    Color('gray', (128, 128, 128)),
]


_shapes = [
    Circle(7.5),
    Rectangle(15, 15),
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images', type=int, default=10000)
    parser.add_argument('--n-non-relational-per-image', type=int, default=10)
    parser.add_argument('--n-relational-per-image', type=int, default=10)
    parser.add_argument('--height', type=int, default=75)
    parser.add_argument('--width', type=int, default=75)
    parser.add_argument('--out', type=str, default='sort_of_clevr.pkl')
    args = parser.parse_args()

    clevr = SortOfCLEVR(
        _background_color, _colors, _shapes, height=args.height,
        width=args.width, n_relational_per_img=args.n_relational_per_image,
        n_non_relational_per_img=args.n_non_relational_per_image)

    images, questions, answers = clevr.generate(args.n_images)

    with open(args.out, 'wb') as f:
        pickle.dump({
            'imgs': images,
            'questions': questions,
            'answers': answers,
            'sort_of_clevr_kwargs': {
                'background_color': clevr.background_color,
                'colors': clevr.colors,
                'shapes': clevr.shapes,
                'height': clevr.height,
                'width': clevr.width,
                'n_relational_per_img': clevr.n_relational_per_img,
                'n_non_relational_per_img': clevr.n_non_relational_per_img,
            },
        }, f)
