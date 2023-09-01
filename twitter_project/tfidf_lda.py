import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


df = pd.read_csv('res_sentiment.csv')
neg_df = df[df['sentiment'] == -1]
pos_df = df[df['sentiment'] == 1]

def topic_analysis(df, n_components, top_n):
    contents = [' '.join(eval(i)) for i in df['tokens'].to_list()]
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    x = tfidf.fit_transform(contents)
    model = LatentDirichletAllocation(n_components=n_components, random_state=42)
    model.fit(x)
    featute_names = tfidf.get_feature_names_out()
    rows = []
    for topic in model.components_:
        topwords = [featute_names[i] for i in topic.argsort()[: -top_n - 1:-1]]
        rows.append(topwords)
    for idx, row in enumerate(rows):
        print(f'topic :{idx + 1}')
        print(row)

print('positive')
topic_analysis(pos_df, 10, 10)
print('negtive')
topic_analysis(neg_df, 10, 10)