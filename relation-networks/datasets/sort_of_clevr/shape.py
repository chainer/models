import numpy as np


class Shape():

    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height

    def draw(self, img, *args, **kwargs):
        raise NotImplementedError


class Rectangle(Shape):

    def __init__(self, width, height):
        super().__init__('rectangle', width, height)

    def draw(self, img, *args, **kwargs):
        img.rectangle(*args, **kwargs)


class Circle(Shape):

    def __init__(self, radius):
        super().__init__('circle', radius * 2, radius * 2)

    def draw(self, img, *args, **kwargs):
        img.ellipse(*args, **kwargs)


class PointsShape(Shape):

    '''Class to define bitmap-like shapes.'''

    def __init__(self, name):
        super().__init__(name, self.template.shape[1], self.template.shape[0])
        ys, xs = np.where(self.template == 1)
        self.points = np.transpose((xs, ys))

    def draw(self, img, *args, **kwargs):
        x0, y0, _, _ = args[0]
        points = self.points.copy()
        points[:, 0] += x0
        points[:, 1] += y0
        img.point(points.ravel().tolist(), **kwargs)


class BoxInABox(PointsShape):

    '''Simple example how to create a bitmap by deriving PointsShape.'''

    template = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    def __init__(self):
        super().__init__('box_in_a_box')
