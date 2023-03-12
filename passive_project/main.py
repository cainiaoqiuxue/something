import os

from method import RegexMethod, PositionMethod


def make_one_txt(path, method_type):
    method = PositionMethod if method_type else RegexMethod
    contents = method.read_contents(path)
    c, p = method.get_passive_sentence(contents)
    method.save_result(c, '.')
    method.save_result(p, '\n')


def make_dir_txts(dir_path, method_type):
    for file in os.listdir(dir_path):
        make_one_txt(os.path.join(dir_path, file), method_type)


if __name__ == '__main__':
    root_dir = './data'
    METHOD = 1
    files = os.listdir(root_dir)
    for file in files:
        file = os.path.join(root_dir, file)
        if os.path.isdir(file):
            make_dir_txts(file, METHOD)
        else:
            make_one_txt(file, METHOD)