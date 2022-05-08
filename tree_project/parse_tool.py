import re


def parse_node(content):
    pattern = '(\d) \\[label=<(.*?) &.*?gini = (0\\.\d+)<br/>samples = (\d+).*?\\]'
    res = re.search(pattern, content)
    if res:
        ans = {
            'id': res.group(1),
            'feature': res.group(2),
            'gini': res.group(3),
            'sample': res.group(4),
        }
        # print(f'id = {res.group(1)}\nfeature = {res.group(2)}\ngini = {res.group(3)}\nsample = {res.group(4)}')
        return ans
    else:
        pattern = '(\d) \\[label=<gini = (0\\.\d+)<br/>samples = (\d+).*?\\]'
        res = re.search(pattern, content)
        if res:
            ans = {
                'id': res.group(1),
                'feature': None,
                'gini': res.group(2),
                'sample': res.group(3),
            }
            # print(f'(leaf node)id = {res.group(1)}\ngini = {res.group(2)}\nsample = {res.group(3)}')
            return ans
        else:
            raise ValueError("parse error")


def parse_connection(content):
    pattern='(\d+) -> (\d+)'
    res=re.search(pattern,content)
    if res:
        return res.group(1),res.group(2)
    else:
        raise ValueError("connection parse error")


def parse(content):
    if not re.match('\d', content[0]):
        return
    if 'gini' in content:
        return parse_node(content)
    else:
        return parse_connection(content)
