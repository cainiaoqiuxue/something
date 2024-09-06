import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

data_dir = Path(__file__) / "../../data"
train = pd.read_csv(data_dir / 'train_data.csv')
dev = pd.read_csv(data_dir / 'dev_data.csv')
test = pd.read_csv(data_dir / 'test_data.csv')
df = pd.concat([train, dev, test], ignore_index=True)
df = df[df['SoH'] != -100].reset_index(drop=True)

def show_displot(data, col):
    sns.displot(data=data, x=col)
    plt.title('Distribution of the col: {}'.format(col))
    plt.show()

def show_relplot(data, col):
    sns.relplot(data, y='SoH', x=col, hue='CS_Name')
    plt.title('Distribution of the col: {}'.format(col))
    plt.show()

# def show_pie(data, col):
#     value_counts = data[col].value_counts()
#     fig, ax = plt.subplots()
#     ax.pie(value_counts, labels=value_counts.index, autopct='%.1f%%', startangle=90)

#     # 设置标题和显示图形
#     ax.set_title('The percent of {}'.format(col))
#     plt.axis('equal')  # 使饼图保持正圆形
#     plt.show()

def show_pie(data, col):
    value_counts = data[col].value_counts()

    # 设置颜色
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    cmap = ListedColormap(colors)

    # 绘制饼图
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(value_counts, autopct='%.1f%%', startangle=90, colors=colors)

    # 设置百分比文本的格式
    plt.setp(autotexts, size=8, weight="bold", color="black")

    # 添加图例
    ax.legend(wedges, value_counts.index, title="CS_Name", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 设置标题和显示图形
    ax.set_title('The percent of {}'.format(col))
    plt.axis('equal')  # 使饼图保持正圆形
    plt.show()

def show_pair(data, cols):
    sns.pairplot(data[cols + ['CS_Name']], hue='CS_Name')
    # plt.title('The pair distribution of the {} and {}'.format(cols[0], cols[1]))
    plt.show()