import re  # 导入正则表达式模块，用于在清洗数据时匹配和处理字符串。
import pandas as pd  # 导入pandas库，是Python数据分析的重要工具，用于数据处理和分析。
import jieba  # 导入jieba库，一个中文分词工具。
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # 从sklearn中导入用于文本特征提取的工具。
from sklearn.decomposition import LatentDirichletAllocation  # 从sklearn中导入LDA(潜在狄利克雷分配)模型，用于主题模型分析。
import torch  # 导入PyTorch，一个深度学习框架。
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification  # 导入transformers库，用于使用预训练模型进行自然语言处理。
from tqdm.auto import tqdm  # 从tqdm库中导入实用工具，用于在循环中显示进度条。
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于数据可视化。

# 设置matplotlib中文支持，通过设置字体避免中文乱码问题。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def sentiment_analysis(df):
    # 从预训练模型加载分词器和情感分析模型。
    tokenizer = AutoTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')
    model = AutoModelForSequenceClassification.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')

    # 将DataFrame中的"text"列转换为列表。
    content = df['text'].tolist()
    # 设置模型运行的设备，优先使用GPU。
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)

    res = []
    with torch.no_grad():  # 关闭梯度计算，因为我们只是在做推理。
        for c in tqdm(content):
            inputs = tokenizer(c, return_tensors='pt')  # 利用分词器对内容进行处理。
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 确保数据在正确的设备上。
            # 模型评估并取得最大对数概率的索引，即预测的情感类别。
            res.append(model(**inputs).logits.cpu().argmax(dim=-1).item())

    # 将分析结果添加到DataFrame中。
    df['情感倾向'] = res
    return df

def plot_sentiment(df):
    # 统计情感倾向并转换为字典。
    data = df['情感倾向'].value_counts().to_dict()
    # 设置情感标签映射。
    sentiment_map = {
        0: '正向情感',
        1: '中立情感',
        2: '负向情感',
    }
    # 获取排序后的值和标签。
    values = data.values()
    labels = [sentiment_map[i] for i in data.keys()]
    # 初始化绘图并设置大小。
    plt.figure(figsize=(10, 8))
    # 绘制饼图。
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=180)
    plt.axis('equal')  # 设置轴比例相等。
    plt.title('情感分析占比')  # 设置图表标题。
    plt.legend()  # 显示图例。
    plt.show()  # 显示图表。


def clean_data(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        # 定义多个正则表达式匹配模式，用于清洗数据。
        '[\t\n\r\f]',
        '[a-zA-Z\d]',
        '[哈啊嘿]',
        '@.*? ',
        '#.*? ',
        '#.*?$',
        # 匹配网址。
        '(http|https|ftp)://...',
    ]
    for pattern in patterns:
        pattern = re.compile(pattern)
        text = re.sub(pattern, ' ', text)  # 逐一使用正则表达式替换匹配文本为空格。
    return text.lower()  # 转换文本为小写。


def split_text(text, stopwords):
    text = jieba.lcut(text)  # 使用结巴分词对文本进行分词。
    res = []
    for word in text:
        # 如果词长度大于1并且不是停用词，添加到结果中。
        if len(word) > 1 and word not in stopwords:
            res.append(word)
    return res

def topic_analysis(tokens, n_components, top_n):
    # 将分词后的列表转换为字符串，以便输入到向量化模型中。
    contents = [' '.join(t) for t in tokens]
    tfidf = TfidfVectorizer()  # 初始化TFIDF向量器。
    x = tfidf.fit_transform(contents)  # 对文本内容进行TFIDF转换。
    # 初始化并训练LDA模型。
    model = LatentDirichletAllocation(n_components=n_components, random_state=42)
    model.fit(x)
    # 获取模型的特征名。
    if hasattr(tfidf, 'get_feature_names_out'):
        feature_names = tfidf.get_feature_names_out()
    else:
        feature_names = tfidf.get_feature_names()
    rows = []
    for topic in model.components_:
        # 获取每个主题的最高概率词汇。
        topwords = [feature_names[i] for i in topic.argsort()[: -top_n - 1:-1]]
        rows.append(topwords)  # 每一行代表一个主题和该主题的关键词。
    return rows

def topic_show(topic_res, title):
    print('{} LDA主题分析关键词提取：'.format(title))
    for i, j in enumerate(topic_res):
        # 打印每个主题的索引和对应的关键词。
        print('主题 {}: {}'.format(
            i + 1,
            ' '.join(j)
        ))