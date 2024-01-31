# -*- coding:utf-8 -*-

# 导入必需的库
import jieba  # 结巴分词库，用于中文文本分词
import matplotlib.pyplot as plt  # 用于绘制图像和图表
from wordcloud import WordCloud  # 生成词云
from collections import Counter  # 提供计数功能，便于统计词频
from tqdm.auto import tqdm  # tqdm库用于显示进度条

# 设置matplotlib的参数，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，以正确显示中文字
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 文本处理工具类
class TextTool:
    # 初始化方法
    def __init__(self, texts, stopwords=None, overwrite=False):
        self.texts = texts  # 将待处理的文本列表保存为实例变量
        jieba.initialize()  # 初始化jieba库
        # 判断是否有停用词列表传入
        if stopwords is None:
            self.stopwords = None
        else:
            # 如果停用词是文件路径，读取文件内容作为停用词表
            if isinstance(stopwords, str):
                with open(stopwords, 'r', encoding='utf-8') as f:
                    stopwords = f.read()
                    stopwords = stopwords.split('\n')
            self.stopwords = set(stopwords)  # 将停用词列表转换为集合以提高查询效率
        self.tokens = None  # 初始化词条统计的变量
        self.overwrite = overwrite  # 是否覆盖先前的词频统计结果

    # 设置文本列表
    def set_texts(self, texts):
        self.texts = texts

    # 对单个文本进行分词
    def cut_text(self, text):
        texts = jieba.lcut(text)  # 使用结巴分词进行分词
        # 如果有停用词，则在分词结果中移除它们
        if self.stopwords:
            texts = [t for t in texts if t not in self.stopwords]
        return texts

    # 获取所有文本的分词结果并统计词频
    def get_tokens(self):
        res = Counter()  # 创建一个Counter对象用于统计
        # 遍历所有文本，对每个文本分词后更新到词频统计中
        for text in tqdm(self.texts):
            res.update(self.cut_text(text))
        # 根据overwrite参数决定是更新还是直接替换分词计数结果
        if self.tokens is None or self.overwrite:
            self.tokens = res
        else:
            self.tokens.update(res)
        # 此行代码已被注释，因为函数不需要返回任何值

    # 绘制词云并显示或保存
    def plot_wordcloud(self, save_path=None):
        plt.figure(figsize=(10, 8))  # 设定绘图尺寸
        word_dic = self.tokens  # 获取分词统计结果
        # 配置词云对象的参数
        wd_config = dict(
            font_path="C:/Windows/Fonts/simsun.ttc",  # 设置字体路径，这里使用宋体
            background_color="white",  # 设置词云背景颜色为白色
            scale=60,  # 设置显示比例
        )
        wd = WordCloud(**wd_config)  # 初始化词云对象
        wd.generate_from_frequencies(word_dic)  # 根据词频生成词云
        plt.imshow(wd)  # 显示词云
        plt.axis('off')  # 不显示坐标轴
        # 如果提供了保存路径，则保存图片，否则显示图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # 绘制出现频率最高的n个词
    def plot_top_n_words(self, n=20, save_path=None):
        top = self.tokens.most_common(n)  # 获取出现频率最高的n个词
        words, count = zip(*top)  # 解压为单独的列表

        plt.figure(figsize=(10, 8))  # 设置绘图尺寸
        plt.bar(words, count)  # 绘制条形图
        plt.xlabel('Words')  # 设置x轴标签
        plt.ylabel('Counts')  # 设置y轴标签
        plt.title(f'Top {n} words')  # 设置图表标题
        plt.xticks(rotation=45)  # 将x轴的标签旋转45度
        plt.tight_layout()  # 自动调整布局
        # 如果有保存路径则保存条形图，否则显示
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

# 文本模型类
class TextModel:
    # 初始化方法
    def __init__(self, model_path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # 从指定路径加载预训练的分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 使用模型预测文本的logits（未归一化的预测值）
    def predict_logits(self, content):
        # 对文本内容分词，将其转换成模型可识别的格式
        inputs = self.tokenizer(content, return_tensors='pt', padding='longest', truncation=True)
        outputs = self.model(**inputs)  # 使用模型进行预测
        return outputs.logits  # 返回logits

    # 对单个内容进行预测，并返回预测类别的索引
    def predict(self, content):
        logits = self.predict_logits(content)  # 获取logits
        return logits.argmax(dim=-1)  # 返回概率最高的类别对应的索引

    # 批量预测多个文本内容，并返回结果列表
    def batch_predict(self, contents):
        res = []  # 初始化结果列表
        for c in contents:  # 遍历所有文本内容
            res.append(self.predict(c).item())  # 对每个内容进行预测，并将结果添加到列表
        return res  # 返回预测结果列表