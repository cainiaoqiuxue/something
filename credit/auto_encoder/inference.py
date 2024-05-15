import os
import numpy as np
import pandas as pd
import joblib

# from auto_encoder.preprocessing import AutoEncoder


def load_scale_model(path):
    return joblib.load(path)


def load_model_weight(path):
    model = AutoEncoder().ae_model(164, 20, 14)
    model.load_weights(path)
    return model


import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#
labels=['违约用户','履约用户']
X=[159610,640390]
#
# plt.pie(X,labels=labels,autopct='%1.2f%%')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["375 g flour",
          "75 g sugar",
          "250 g butter",
          "300 g berries"]

# data = [float(x.split()[0]) for x in recipe]
data=X
# ingredients = [x.split()[-1] for x in recipe]
ingredients = labels


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.2f}%\n({:d} 人)".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  explode=(0,0.1),shadow=True,
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="",
          loc="upper left",
          fontsize='xx-large',
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=20, weight="bold")

# ax.set_title("Matplotlib bakery: A pie")

plt.show()