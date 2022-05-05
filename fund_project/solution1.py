import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_frame(kind):
    with open(f'./data/{kind}_funds_data.txt', 'r') as f:
        data = f.readlines()
    df = pd.DataFrame([json.loads(i) for i in data])
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: np.NaN if x.strip() == '--' else float(x[:-1]))
    df['kind'] = df['kind'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    return df


etf = make_frame('ETF')
fof = make_frame('FOF')
lof = make_frame('LOF')

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(16, 8))
size = 0.3
cmap = plt.colormaps["tab20c"]
outer_colors = cmap(np.arange(3) * 4)
inner_colors = cmap([1, 2, 5, 6, 9, 10])
etfv = etf['kind'].value_counts()
lofv = lof['kind'].value_counts()
w = plt.pie(lofv, radius=1, colors=outer_colors,
            # autopct='%.2f%%',pctdistance=0.85,
            wedgeprops=dict(width=size, edgecolor='w'))
plt.pie(etfv, radius=1 - size, colors=inner_colors,
        # autopct='%.2f%%',pctdistance=0.85,
        wedgeprops=dict(width=size, edgecolor='w'))
plt.legend(w[0], lofv.index, loc='best')
plt.title('investment type of LOF & ETF')
plt.tight_layout()

times = ['1w', '1m', '3m', '6m', 'ty', '1y', '2y', '3y']
etft = []
loft = []
for time in times:
    etft.append((etf[f'nav_{time}'] > etf[f'nav_{time}_mean']).mean())
    loft.append((lof[f'nav_{time}'] > lof[f'nav_{time}_mean']).mean())
plt.figure(figsize=(16, 8))
index = np.arange(len(times))
bar_width = 0.36
plt.bar(index, etft, width=bar_width, label='etf')
plt.bar(index + bar_width, loft, width=bar_width, label='lof')
plt.xlabel('return')
plt.ylabel('fraction')
plt.xticks(index + bar_width / 2, ['近1周', '近1月', '近3月', '近6月', '今年来', '近1年', '近2年', '近3年'])
plt.title('Fraction of funds performing better than category average')
plt.legend()

plt.figure(figsize=(16, 8))
times = ['1y', '2y', '3y']
index = np.arange(3)
etf_std = []
etf_sharp = []
lof_std = []
lof_sharp = []
for time in times:
    etf_std.append((etf[f'std_{time}'] > etf[f'std_{time}'].mean()).mean())
    lof_std.append((lof[f'std_{time}'] > lof[f'std_{time}'].mean()).mean())
    etf_sharp.append((etf[f'sharp_{time}'] > etf[f'sharp_{time}'].mean()).mean())
    lof_sharp.append((lof[f'sharp_{time}'] > lof[f'sharp_{time}'].mean()).mean())
ax1 = plt.subplot(1, 2, 1)
plt.title('Fraction of ETF with higher \nstandard deviation and sharpe ratio than category')
plt.bar(index, etf_std, width=bar_width, label='etf_std')
plt.bar(index + bar_width, etf_sharp, width=bar_width, label='etf_sharp')
plt.legend()
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.title('Fraction of LOF with higher \nstandard deviation and sharpe ratio than category')
plt.bar(index, lof_std, width=bar_width, label='lof_std')
plt.bar(index + bar_width, lof_sharp, width=bar_width, label='lof_sharp')
plt.legend()
plt.show()
