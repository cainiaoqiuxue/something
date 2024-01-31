# -*- coding:utf-8 -*-
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from tqdm.auto import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TextTool:
    def __init__(self, texts, stopwords=None, overwrite=False):
        self.texts = texts
        jieba.initialize()
        if stopwords is None:
            self.stopwords = None
        else:
            if isinstance(stopwords, str):
                with open(stopwords, 'r', encoding='utf-8') as f:
                    stopwords = f.read()
                    stopwords = stopwords.split('\n')
            self.stopwords = set(stopwords)
        self.tokens = None
        self.overwrite = overwrite

    def set_texts(self, texts):
        self.texts = texts

    def cut_text(self, text):
        texts = jieba.lcut(text)
        if self.stopwords:
            texts = [t for t in texts if t not in self.stopwords]
        return texts

    def get_tokens(self):
        res = Counter()
        for text in tqdm(self.texts):
            res.update(self.cut_text(text))
        if self.tokens is None or self.overwrite:
            self.tokens = res
        else:
            self.tokens.update(res)
        # return res

    def plot_wordcloud(self, save_path=None):
        plt.figure(figsize=(10, 8))
        word_dic = self.tokens
        wd_config = dict(
            font_path="C:/Windows/Fonts/simsun.ttc",
            background_color="white",
            scale=60,
        )
        wd = WordCloud(**wd_config)
        wd.generate_from_frequencies(word_dic)
        plt.imshow(wd)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_top_n_words(self, n=20, save_path=None):
        top = self.tokens.most_common(n)
        words, count = zip(*top)

        plt.figure(figsize=(10, 8))
        plt.bar(words, count)
        plt.xlabel('Words')
        plt.ylabel('Counts')
        plt.title(f'Top {n} words')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


class TextModel:
    def __init__(self, model_path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict_logits(self, content):
        inputs = self.tokenizer(content, return_tensors='pt', padding='longest', truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits

    def predict(self, content):
        logits = self.predict_logits(content)
        return logits.argmax(dim=-1)

    def batch_predict(self, contents):
        res = []
        for c in contents:
            res.append(self.predict(c).item())
        return res