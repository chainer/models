import numpy as np
from PIL import Image
from PIL import ImageDraw


_relational_questions = [
    'What is the shape of the object that is closes to the {color_name} object?',  # NOQA
    'What is the shape of the object that is furthest from the {color_name} object?',  # NOQA
    'How many objects have the shape of the {color_name} object?',  # NOQA
]


_non_relational_questions = [
    'What is the shape of the {color_name} object?',  # NOQA
    'Is the {color_name} object on the left or right of the image?',  # NOQA
    'Is the {color_name} object on the top or bottom of the image?',  # NOQA
]


class Object(object):

    def __init__(self, x0, y0, x1, y1, color, shape):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.color = color
        self.shape = shape


def create_vocab(shapes, colors):
    '''Creates an answer vocabulary.'''
    vocab = ['left', 'right', 'top', 'bottom']
    vocab += [s.name for s in shapes]
    vocab += [c.name for c in colors]
    vocab += [str(i) for i in range(1, len(colors) + 1)]
    return vocab


class SortOfCLEVR(object):

    '''Sort-of-CLEVR dataset generator.'''

    def __init__(
            self, background_color, colors, shapes, height=75, width=75,
            n_relational_per_img=10, n_non_relational_per_img=10, mode='RGB'):
        self.background_color = background_color
        self.colors = colors
        self.shapes = shapes
        self.height = height
        self.width = width
        self.n_relational_per_img = n_relational_per_img
        self.n_non_relational_per_img = n_non_relational_per_img
        self.mode = mode
        self.n_objects = len(colors)

        # A question consists of a target object identified by its colors, a
        # question type (out of two; non-relational or relational) and a
        # sub-question type (out of three).
        self.question_length = len(colors) + 2 + 3

        self.vocab = create_vocab(shapes, colors)

        for i, shape in enumerate(shapes):
            shape.id = i

    def generate(self, n_imgs):
        '''Generate n_imgs random images with questions.'''
        n_rel = self.n_relational_per_img
        n_non_rel = self.n_non_relational_per_img

        imgs = np.empty(
            (n_imgs, 3, self.height, self.width), dtype=np.float32)
        questions = np.empty(
            (n_imgs, n_non_rel + n_rel, self.question_length), dtype=np.int32)
        answers = np.empty(
            (n_imgs, n_non_rel + n_rel), dtype=np.int32)

        for i in range(n_imgs):
            imgs[i], questions[i], answers[i] = self._generate_img()

        return imgs, questions, answers

    def _generate_img(self):
        img = Image.new(
            self.mode, (self.height, self.width), self.background_color.rgb)
        img_draw = ImageDraw.Draw(img)

        # Generate a random object per color.
        objs = self._generate_objects()

        for obj in objs:
            obj.shape.draw(
                img_draw, (obj.x0, obj.y0, obj.x1, obj.y1),
                fill=tuple(obj.color.rgb))

        # Generate relational and non-relational questions.
        questions, answers = self._generate_question_answers(objs)

        img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        questions = np.asarray(questions, dtype=np.int32)
        answers = np.asarray(answers, dtype=np.int32)

        return img, questions, answers

    def decode_question(self, question):
        '''Returns a decoded human-readable question.'''
        # Decode the object and its color.
        n_objs = self.n_objects
        obj_i = np.where(question[:n_objs] == 1)[0][0]
        color_name = self.colors[obj_i].name

        # Decode the question and sub-question types.
        question_type = np.where(question[n_objs:n_objs + 2] == 1)[0][0]
        question_sub_type = np.where(question[n_objs + 2:] == 1)[0][0]

        if question_type == 0:
            decoded = _relational_questions[question_sub_type]
        elif question_type == 1:
            decoded = _non_relational_questions[question_sub_type]
        else:
            raise RuntimeError

        return decoded.format(color_name=color_name)

    def decode_answer(self, answer_token):
        ''' Returns a decoded human-readable answer.'''
        return self.vocab[answer_token]

    def _generate_objects(self):
        objs = []

        def _find_non_overlapping_position(shape):
            tries = 0
            max_tries = 128

            while True:
                w = shape.width
                h = shape.height
                x0 = np.random.randint(0, self.width - w + 1)
                y0 = np.random.randint(0, self.height - h + 1)
                x1 = x0 + w
                y1 = y0 + h
                is_overlapping = False

                for obj in objs:
                    if (max(x0, obj.x0) > min(x1, obj.x1) or
                            max(y0, obj.y0) > min(y1, obj.y1)):
                        continue  # No intersection.
                    else:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    return x0, y0, x1, y1

                tries += 1
                if tries >= max_tries:
                    raise RuntimeError(
                        'Failed to generate non-overlapping random shapes.')

        for color in self.colors:
            shape = np.random.choice(self.shapes)
            position = _find_non_overlapping_position(shape)
            objs.append(Object(*position, color, shape))

        return objs

    def _generate_question_answers(self, objs):
        # Asserts that the objects are given in the same order as the colors.
        questions = []
        answers = []

        def _random_color():
            return np.random.randint(0, self.n_objects)

        def _random_sub_question():
            assert len(_relational_questions) == len(_non_relational_questions)
            return np.random.randint(0, len(_relational_questions))

        n_directions = 4  # Left, right, top, bottom.
        n_colors = len(self.colors)
        n_shapes = len(self.shapes)

        for i in range(
                self.n_relational_per_img + self.n_non_relational_per_img):
            color_i = _random_color()
            question_i = i // self.n_relational_per_img  # 0 or 1 by default.
            sub_question_i = _random_sub_question()  # 0, 1 or 2 by default.

            question = np.zeros(self.question_length)
            question[color_i] = 1
            question[n_colors + question_i] = 1
            question[n_colors + 2 + sub_question_i] = 1

            obj = objs[color_i]

            if question_i == 0:  # Relational question.
                if sub_question_i == 0:
                    closest_obj = None
                    closest_dist = self.height * self.width
                    for i, other in enumerate(objs):
                        if i == color_i:
                            continue  # Skip self.
                        dist = np.linalg.norm([
                            (other.x0 - obj.x0), (other.y0 - obj.y0)])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_obj = other
                    answer = n_directions + closest_obj.shape.id
                elif sub_question_i == 1:
                    furthest_obj = None
                    furthest_dist = 0
                    for i, other in enumerate(objs):
                        if i == color_i:
                            continue  # Skip self.
                        dist = np.linalg.norm([
                            (other.x0 - obj.x0), (other.y0 - obj.y0)])
                        if dist > furthest_dist:
                            furthest_dist = dist
                            furthest_obj = other
                    answer = n_directions + furthest_obj.shape.id
                else:
                    count = -1  # At least one increment to 0 with itself.
                    for i, other in enumerate(objs):
                        if other.shape.name == obj.shape.name:
                            count += 1
                    answer = n_directions + n_shapes + n_colors + count
            else:  # Non-relational question.
                if sub_question_i == 0:
                    assert obj.color == self.colors[color_i]
                    answer = n_directions + obj.shape.id  # Shape.
                elif sub_question_i == 1:
                    left = obj.x0 + ((obj.x1 - obj.x0) / 2) < (self.width / 2)
                    answer = 0 if left else 1
                else:
                    top = obj.y0 + ((obj.y1 - obj.y0) / 2) < (self.height / 2)
                    answer = 2 if top else 3

            assert sum(question) == 3
            questions.append(question)
            answers.append(answer)

        return questions, answers
