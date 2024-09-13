import pandas as pd
from pathlib import Path
from process import Process


class Docker:
    def __init__(self, idx, cap=0):
        self.idx = idx
        self.rows, self.row_idx, self.cols = self.init_docker()
        self.row = self.rows[self.idx]
        self.row_id = self.row_idx[self.idx]
        self.container = []
        self.cap = cap

    def init_docker(self):
        rows = '8 8 8 7 16 17 17 17 17 17 17 17 17 17 17 17 17 17 17 15 15 17 5'.split(' ')
        rows = list(map(int, rows))
        row_idx = list(range(13, 36))
        cols = '20 22 24 26 28 32 34 36 38 40 44 46 48 50 52 54 56 58'.split(' ')
        cols = list(map(int, cols))
        return rows, row_idx, cols
    
    def add(self, goods):
        self.container.append(goods)

    @property
    def total(self):
        tol = 0
        for g in self.container:
            tol += g.weight
        return tol
    
    def show_info(self):
        result = {}
        for g in self.container:
            result[g.name] = self.row_id
        return result


class Goods:
    def __init__(self, name, weight, height):
        self.name = name
        self.weight = weight
        self.height = height


class GoodSequence:
    def __init__(self, df, docker_num=23):
        self.df = df
        self.docker_num = docker_num
        self.dockers = [Docker(i) for i in range(self.docker_num)]
        self.goods = self.make_goods()

    def cal_mean_weight(self):
        total_weight = self.df['weight'].sum()
        bins = sum(self.dockers[0].rows)
        mean_weight = total_weight / bins
        for i in range(self.docker_num):
            self.dockers[i].cap = mean_weight * self.dockers[i].row

    def make_goods(self):
        self.df = self.df.sort_values('weight')
        goods = []
        for i, x in self.df.iterrows():
            goods.append(Goods(x['name'], x['weight'], x['height']))
        return goods
    
    def forward(self, thr):
        cur_docker = 0
        cur_good_idx = 0
        while cur_good_idx < len(self.goods):
            cur_good = self.goods[cur_good_idx]
            if self.dockers[cur_docker].total + cur_good.weight <= self.dockers[cur_docker].cap:
                self.dockers[cur_docker].add(cur_good)
                cur_good_idx += 1
            elif self.dockers[cur_docker].total + cur_good.weight < self.dockers[cur_docker].cap + thr:
                self.dockers[cur_docker].add(cur_good)
                cur_docker += 1
                cur_good_idx += 1
            else:
                cur_docker += 1

    def generate_dict(self):
        result = dict()
        for docker in self.dockers:
            result.update(docker.show_info())
        return result


def docker_main():
    p = Process()
    p.clean_data()
    weights = pd.read_csv(p.data_root / 'result1.csv', dtype=str)
    volumns = pd.read_csv(p.data_root / 'Volume.csv', dtype=str)
    weights = weights[weights['label'] != 'low'][p.id]
    p.df = p.df[p.df[p.id].isin(weights)].reset_index(drop=True)
    tmp = pd.merge(p.df.groupby(p.id).first().reset_index(), volumns[[p.id, 'Gross Cubic Dim', 'Gross Weight']], on=p.id, how='left') 
    tmp = tmp.drop_duplicates(subset=[p.id]).dropna(subset=['Gross Cubic Dim', 'Gross Weight'])
    tmp = tmp[[p.id, 'Gross Cubic Dim', 'Gross Weight']]
    tmp['Gross Cubic Dim'] = tmp['Gross Cubic Dim'].str.replace(',', '').astype(float)
    tmp['Gross Weight'] = tmp['Gross Weight'].str.replace(',', '').astype(float)

    large = tmp[(tmp['Gross Weight'] > 10) | (tmp['Gross Cubic Dim'] > 6000)]
    small = tmp[~tmp[p.id].isin(large[p.id])]
    large.columns = ['name', 'height', 'weight']
    small.columns = ['name', 'height', 'weight']

    gs = GoodSequence(large)
    gs.cal_mean_weight()
    # gs.forward(200)
    gs.forward(300)
    large_result = gs.generate_dict()
    large_df = pd.DataFrame({'name': large_result.keys(), 'row': large_result.values()})
    large_df['type'] = 'large'

    gs = GoodSequence(small)
    gs.cal_mean_weight()
    # gs.forward(2)
    gs.forward(1)
    small_result = gs.generate_dict()
    small_df = pd.DataFrame({'name': small_result.keys(), 'row': small_result.values()})
    small_df['type'] = 'small'

    result = pd.concat([large_df, small_df], ignore_index=True)
    result.columns = [p.id, 'row_id', 'type']
    return result


if __name__ == '__main__':
    result = docker_main()
    result.to_csv('result2.csv', index=False)