import os
import re

from nltk import pos_tag

from config import Config
from utils import read_contents, save_to_txt


class RegexMethod(object):
    cur_path = None
    method_name = 'regex'

    @classmethod
    def read_contents(cls, path):
        contents = read_contents(path)
        cls.cur_path = path
        return contents

    @staticmethod
    def get_passive_sentence(contents):
        complete_res = []
        passive_res = []
        for content in contents:
            g = re.search(Config.passive_pattern, content)
            if g:
                passive_res.append(content)
                complete_res.append(Config.start_token + content + Config.end_token)
            else:
                complete_res.append(content)
        return complete_res, passive_res

    @classmethod
    def save_result(cls, contents, sep, path=None):
        save_dir, file_name = os.path.split(cls.cur_path)
        content = sep.join(contents)
        sep_idx = {'.': '_complete_regex_content', '\n': '_compassive_regex_content'}
        file_name = os.path.splitext(file_name)[0] + sep_idx.get(sep, '_content') + os.path.splitext(file_name)[1]
        if path is None:
            res_dir = os.path.join(save_dir, 'result')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            path = os.path.join(res_dir, file_name)
        save_to_txt(path, content)


class PositionMethod(object):
    cur_path = None
    method_name = 'position'

    @classmethod
    def read_contents(cls, path):
        contents = read_contents(path)
        cls.cur_path = path
        return contents

    @staticmethod
    def check_pos_sentence(content):
        try:
            content = content.split(' ')
            res = pos_tag(content)
            res = [r[1] for r in res]
            return 'VBN' in res
        except:
            return False

    @classmethod
    def get_passive_sentence(cls, contents):
        complete_res = []
        passive_res = []
        for content in contents:
            g = cls.check_pos_sentence(content)
            if g:
                passive_res.append(content)
                complete_res.append(Config.start_token + content + Config.end_token)
            else:
                complete_res.append(content)
        return complete_res, passive_res

    @classmethod
    def save_result(cls, contents, sep, path=None):
        save_dir, file_name = os.path.split(cls.cur_path)
        content = sep.join(contents)
        sep_idx = {'.': '_complete_pos_content', '\n': '_compassive_pos_content'}
        file_name = os.path.splitext(file_name)[0] + sep_idx.get(sep, '_content') + os.path.splitext(file_name)[1]
        if path is None:
            res_dir = os.path.join(save_dir, 'result')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            path = os.path.join(res_dir, file_name)
        save_to_txt(path, content)