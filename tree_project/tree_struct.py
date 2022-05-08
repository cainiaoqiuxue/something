from parse_tool import parse


class TreeNode(object):
    def __init__(self, id, feature, gini, sample):
        self.id = id
        self.gini = float(gini)
        self.sample = int(sample)
        self.depth = 0
        if feature:
            self.feature = feature
            self.leaf = False
        else:
            self.feature = None
            self.leaf = True
        self.left = None
        self.right = None


class CARTNode(object):
    def __init__(self, contents):
        self.node_list = []
        self.contents = contents

    def parse_content(self):
        for content in self.contents.split('\n'):
            res = parse(content)
            if isinstance(res, dict):
                self.node_list.append(TreeNode(**res))
            elif isinstance(res, tuple):
                father = int(res[0])
                child = int(res[1])
                if not self.node_list[father].left:
                    self.node_list[father].left = self.node_list[child]
                else:
                    self.node_list[father].right = self.node_list[child]
                self.node_list[child].depth = self.node_list[father].depth + 1

    def show_info(self):
        if not self.node_list:
            self.parse_content()
        for node in self.node_list:
            print(
                f'id:{node.id} feature:{node.feature} gini:{node.gini} sample:{node.sample} depth:{node.depth} leaf:{node.leaf}')
            print(f'left:{node.left.id if node.left else None},right:{node.right.id if node.right else None}')
            print('---' * 20)

    def cal_acc(self, normalize=True):
        if not self.node_list:
            self.parse_content()
        res = dict()
        tol = self.node_list[0].sample
        for node in self.node_list:
            if node.feature:
                if node.feature not in res:
                    res[node.feature] = 0
                temp = node.gini * node.sample
                temp -= node.left.gini * node.left.sample
                temp -= node.right.gini * node.right.sample
                res[node.feature] += temp / tol
        if normalize:
            s = sum(res.values())
            for key in res:
                res[key] = res[key] / s
        return res

    def cal_change_acc(self, normalize=True):
        if not self.node_list:
            self.parse_content()
        res = dict()
        tol = self.node_list[0].sample
        max_depth = 0
        for node in self.node_list:
            max_depth = max(max_depth, node.depth)
        for node in self.node_list:
            if node.feature:
                if node.feature not in res:
                    res[node.feature] = 0
                temp = node.gini * node.sample
                temp -= node.left.gini * node.left.sample
                temp -= node.right.gini * node.right.sample
                weight = (max_depth - node.depth) / max_depth
                res[node.feature] += (temp / tol) * weight
        if normalize:
            s = sum(res.values())
            for key in res:
                res[key] = res[key] / s
        return res