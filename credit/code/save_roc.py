import numpy as np
import matplotlib.pyplot as plt

lr_fpr=np.load('../save_model/lr_fpr.npy')
lr_tpr=np.load('../save_model/lr_tpr.npy')

svm_fpr=np.load('../save_model/svm_fpr.npy')
svm_tpr=np.load('../save_model/svm_tpr.npy')

xgb_fpr=np.load('../save_model/xgb_fpr.npy')
xgb_tpr=np.load('../save_model/xgb_tpr.npy')

rf_fpr=np.load('../save_model/rf_fpr.npy')
rf_tpr=np.load('../save_model/rf_tpr.npy')


rfi_fpr=np.load('../save_model/rf_im_fpr.npy')
rfi_tpr=np.load('../save_model/rf__im_tpr.npy')


plt.plot(lr_fpr,lr_tpr,'--',label='lr')
plt.plot(svm_fpr,svm_tpr,'.-',label='svm')
plt.plot(xgb_fpr,xgb_tpr,':',label='xgb')
plt.plot(rf_fpr,rf_tpr,'x-',label='rf')
plt.plot(rfi_fpr,rfi_tpr,label='rf_improve')
plt.legend()
plt.show()