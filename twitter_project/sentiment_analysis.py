import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

def predict(text):
    token = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**token).logits
    return output.argmax().item() - 1

df = pd.read_csv('../input/twitter-spider/res.csv')
contents = df['clean_data'].tolist()
labels = []
for i in tqdm(contents):
    labels.append(predict(str(i)))
df['sentiment'] = labels
df.to_csv('res_sentiment.csv', index=False)