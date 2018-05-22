from bs4 import BeautifulSoup
import os

def parse_samples(path, parsed_list, y, type_of_sample):
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if full_path.endswith('.html'):
            s = get_text_from_html(full_path)
            parsed_list.append(s)
            y.append(type_of_sample)

def get_text_from_html(path):
    f = open(path)
    soup = BeautifulSoup(f, 'html.parser')
    table = soup.find_all('p')
    s = ''
    p = []
    for x in table:
        if(x.text != 'placeholder'):
            s += x.text + ' '
    f.close()
    s = ''.join([i for i in s if not i.isdigit()])
    return s