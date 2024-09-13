import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from process import Process
from docker import Docker, docker_main

class Optimize:
    def __init__(self, p: Process):
        docker = Docker(0)
        self.p = p
        self.rows = docker.rows
        self.cols = docker.cols
        self.center = [32, 34, 36, 38, 40]

    def get_result_1(self):
        self.p.clean_data()
        wd = self.p.cal_proior()
        result1 = self.p.df[[self.p.id, self.p.sid, 'label']].groupby(self.p.id).first().reset_index()
        return result1
    
    def get_result_2(self):
        result2 = docker_main()
        # result2['row'] = result2['row_id'] - 13
        return result2
    
    def get_result(self):
        result1 = self.get_result_1()
        result2 = self.get_result_2()
        result = pd.merge(result2, result1, on=self.p.id, how='left')
        return result
    
    def get_association(self, data, min_support=0.005, top=20):
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)].sort_values('support', ascending=False)
        result = frequent_itemsets.iloc[:top]['itemsets'].values
        result = [[*i] for i in result]
        return result
    
    def get_docker_row_col(self, row_id):
        idx = row_id - 13
        row = self.rows[idx]
        cols = self.cols[:row]
        return row, cols

    def forward(self):
        location = {}
        result = self.get_result()
        row_ids = result['row_id'].unique()
        for row_id in row_ids:
            tmp = result[result['row_id'] == row_id]
            row, cols = self.get_docker_row_col(row_id)
            centers = []
            corners = []
            for col in cols:
                if col in self.center:
                    centers.append(col)
                else:
                    corners.append(col)
            high_count = 0
            medium_count = 0
            for _, x in tmp.iterrows():
                if x['label'] == 'High':
                    if len(centers) > 0:
                        idx = high_count % len(centers)
                        location[x[self.p.id]] = (row_id, centers[idx])
                    else:
                        idx = high_count % len(corners)
                        location[x[self.p.id]] = (row_id, corners[idx])
                    high_count += 1
                else:
                    idx = medium_count % len(corners)
                    location[x[self.p.id]] = (row_id, corners[idx])
                    medium_count += 1
        return location
    
    def forward_with_association(self):
        location = self.forward()
        data = self.p.df.groupby(self.p.key)[self.p.id].agg(list)
        it = self.get_association(data)

        changes = set()
        for a, b in it:
            if a in changes and b in changes:
                continue
            elif a in changes:
                location[b] = (location[b][0], location[a][1])
                changes.add(b)
            elif b in changes:
                location[a] = (location[a][0], location[b][1])
            else:
                if location[a][1] in self.center:
                    location[b] = (location[b][0], location[a][1])
                else:
                    location[a] = (location[a][0], location[b][1])
                changes.add(a)
                changes.add(b)
        return location

if __name__ == '__main__':
    p = Process()
    op = Optimize(p)
    location = op.forward_with_association()
    pd.DataFrame(location.items(), columns=[p.id, 'location']).to_csv('result3.csv', index=False)