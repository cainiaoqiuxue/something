import re

from config import Config


def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        res = f.read()
    return res


def clean_content(content):
    for key, value in Config.clean_patterns.items():
        content = re.sub(key, value, content)
    return content.strip()


def split_content(content):
    contents = re.split(Config.split_pattern, content)
    return contents


def find_ab_re(content):
    begin = re.search('Abstract|abstract|ABSTRACT|A B S T R A C T', content)
    end = re.search('Reference|REFERENCE|Bibliography', content)
    if begin and end:
        return content[begin.span()[1]:end.span()[0]]
    else:
        return content


def read_contents(path):
    print('read txt file: {}'.format(path))
    content = read_txt_file(path)
    content = find_ab_re(content)
    contents = split_content(content)
    contents = [clean_content(i) for i in contents if len(i) > Config.drop_length]
    print('done')
    return contents


def save_to_txt(path, content):
    print('save result to: {}'.format(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('done')
