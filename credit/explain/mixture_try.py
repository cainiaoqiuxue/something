# import xgboost as xgb
# import pandas as pd
# import matplotlib.pyplot as plt

# model=xgb.XGBClassifier()
# model.load_model('../save_model/xgb_model.json')

# df=pd.read_csv('../data/tiny_data_preprocess_v1.csv')
# target=df.pop('isDefault')

# model.predict(df)
# xgb.plot_importance(model,max_num_features=20)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# df=pd.read_csv('data/tiny_train.csv')
# df=df[['grade','isDefault']]


# from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
# import matplotlib.pyplot as plt
# import pandas as pd


# fig = plt.figure()

# host = fig.add_axes([0.15, 0.1, 0.65, 0.8],axes_class=HostAxes)
# par1 = ParasiteAxes(host, sharex=host)
# host.parasites.append(par1)

# host.axis["right"].set_visible(False)

# par1.axis["right"].set_visible(True)
# par1.axis["right"].major_ticklabels.set_visible(True)
# par1.axis["right"].label.set_visible(True)

# df=pd.read_csv('data/train.csv')
# df=df[['grade','isDefault','id']]

# df=df.groupby(['grade','isDefault']).count()

# grade=['A','B','C','D','E','F']
# weiyue=[df.loc[(i,1)].item() for i in grade]
# not_weiyue=[df.loc[(i,0)].item() for i in grade]

# width=0.35
# host.bar(grade,not_weiyue,width,label='buweiyue')
# host.bar(grade,weiyue,width,label='weiyue',bottom=not_weiyue)

# par1.plot(range(len(not_weiyue)),[weiyue[i]/(weiyue[i]+not_weiyue[i]) for i in range(len(not_weiyue))],'.-')
# par1.set_ylim(-0.5,0.5)


# host.set_xlabel("Distance")
# host.set_ylabel("Density")
# par1.set_ylabel("Temperature")

# host.legend()

# print(df)
# plt.show()



# import pandas as pd
# from sklearn.model_selection import train_test_split
# df=pd.read_csv('data/train.csv')

# df=df[['grade','subGrade','isDefault']]
# # print(df['grade'].value_counts())
# # print(df['subGrade'].value_counts())

# df=df.groupby(['subGrade','isDefault']).count()
# print(df.head(35))
# print(df.tail(35))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('data/tiny_train.csv')
df=df[['isDefault','dti']]

sns.kdeplot('dti',data=df[df['isDefault']==1],shade=True,legend=True,label='weiyue')
sns.kdeplot('dti',data=df[df['isDefault']==0],shade=True,legend=True,label='lvyue')
plt.legend()
plt.show()