import matplotlib.pyplot as plt
import numpy as np


fpr=np.load('../save_model/lr_fpr.npy')
tpr=np.load('../save_model/lr_tpr.npy')

plt.plot(fpr,tpr,label='ROC')
plt.fill(np.append(fpr,1),np.append(tpr,0),alpha=0.25,label='AUC')
plt.legend()
plt.show()