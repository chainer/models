import pickle

from chainer import dataset

from datasets.sort_of_clevr.sort_of_clevr import SortOfCLEVR


class SortOfCLEVRDataset(dataset.DatasetMixin):

    '''Sort-of-CLEVR data reader.'''

    def __init__(self, imgs, questions, answers):
        n_imgs = imgs.shape[0]
        if n_imgs != questions.shape[0]:
            raise ValueError
        if n_imgs != answers.shape[0]:
            raise ValueError

        self.imgs = imgs
        self.questions = questions
        self.answers = answers
        self.n_questions_per_img = questions.shape[1]
        self.tot_n_questions = n_imgs * self.n_questions_per_img

    def __len__(self):
        return self.tot_n_questions

    def get_example(self, i):
        img_i = i // self.n_questions_per_img
        question_i = i % self.n_questions_per_img

        img = self.imgs[img_i]
        question = self.questions[img_i, question_i]
        answer = self.answers[img_i, question_i]

        return img, question, answer


def get_sort_of_clevr(sort_of_clevr_path):
    with open(sort_of_clevr_path, 'rb') as f:
        data = pickle.load(f)

    dataset = SortOfCLEVRDataset(
        data['imgs'], data['questions'], data['answers'])
    clevr = SortOfCLEVR(**data['sort_of_clevr_kwargs'])
    return dataset, clevr
