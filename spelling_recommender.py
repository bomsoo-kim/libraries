#ref) https://www.nltk.org/_modules/nltk/metrics/association.html#NgramAssocMeasures
#ref) https://www.nltk.org/api/nltk.metrics.html#nltk.metrics.association.NgramAssocMeasures.jaccard
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance # https://en.wikipedia.org/wiki/Jaccard_index
from nltk.metrics.distance import edit_distance # https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance

def spelling_recommender(word, correct_spellings, len_word_margin = 1):
    len_word = len(word)
#     w_dist = [(w, jaccard_distance(set(word), set(w))) for w in correct_spellings if word[0] == w[0] and abs(len_word - len(w)) <= len_word_margin]
    w_dist = [(w, edit_distance(word, w, transpositions = True)) for w in correct_spellings if word[0] == w[0] and abs(len_word - len(w)) <= len_word_margin]
    return min(w_dist, key = lambda x: x[1])

correct_spellings = words.words()
print('total # of words = ', len(correct_spellings))

#### Getting Started! ##############################################
word = 'cormulent' # incorrect spelling
# word = 'incendenece' # incorrect spelling
# word = 'validrate' # incorrect spelling

word_recommend = spelling_recommender(word, correct_spellings)
word_recommend
