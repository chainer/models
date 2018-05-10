import sys
from bilm import split_sentence_with_punctuations


for l in sys.stdin:
    print(' '.join(split_sentence_with_punctuations(l)))
