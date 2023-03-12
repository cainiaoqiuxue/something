import os


class Config:
    root_dir = os.path.dirname(os.path.abspath(__file__))

    start_token = '[START]'
    end_token = '[END]'
    drop_length=5

    split_pattern = '\.|\?'
    clean_patterns = {
        '\\n': ' ',
        '\\x02': '',
    }

    passive_pattern = '.*?(am|is|are|was|were) \w+ed .*?'