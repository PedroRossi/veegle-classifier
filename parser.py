from bs4 import BeautifulSoup
from string import punctuation
import nltk
import os

nltk.download('stopwords')

def parse_samples(path, parsed_list, y, type_of_sample):
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if full_path.endswith('.html'):
            s = get_text_from_html(full_path)
            parsed_list.append(s)
            y.append(type_of_sample)

def get_text_from_html(path, remove_stopwords = False):
    f = open(path)
    soup = BeautifulSoup(f, 'html.parser')
    p_tags = soup.find_all('p')
    s = []
    for x in p_tags:
        s += x.text.split(' ')
    for i in range(1,6):
        h_tags = soup.find_all('h'+str(i))
        for x in h_tags:
            s += x.text.split(' ')
    s += soup.title.string.split(' ')
    f.close()
    dictionary = list(punctuation)
    dictionary.append('placeholder')
    if remove_stopwords:
        dictionary += nltk.corpus.stopwords.words('portuguese')
    l = [i for i in s if not i.isdigit() and i not in dictionary]
    s = ''.join(l)
    return s