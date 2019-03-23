import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


mu=np.zeros((2,1))
sigma=np.zeros((2,1))

def estimate_gaussian_parameters(Xtrain):
    mu[0] = np.mean(Xtrain[:,0])
    mu[1] = np.mean(Xtrain[:,1])

    sigma[0] = np.std(Xtrain[:,0])
    sigma[1] = np.std(Xtrain[:,1])
    
    return mu,sigma

def prob_distribution_train(p_train,X):
    p_train[:,0] = stats.norm(mu[0],sigma[0]).pdf(X[:,0])
    p_train[:,1] = stats.norm(mu[1],sigma[1]).pdf(X[:,1])
    
    p_train_total = p_train[:,0]*p_train[:,1]
    return p_train_total

  
def prob_distribution_val(p,Xval,Yval):
    p[:,0] = stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])
    p[:,1] = stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])
    
    p_total = (p[:,0]*p[:,1]).reshape(-1,1)
    return p_total
    
def select_threshold(pval,Yval):
    best_epsilon=0
    best_f1=0
    f1=0
    step = (pval.max()-pval.min())/1000
    for epsilon in np.arange(pval.min(),pval.max(),step):
        preds  = (pval < epsilon).reshape(-1,1)
        total_pred = np.hstack((preds,Yval))
        tp = np.sum(np.logical_and(total_pred[:,0]==1,total_pred[:,1] ==1)).astype(float)
        fn = np.sum(np.logical_and(total_pred[:,0]==0,total_pred[:,1])).astype(float)
        fp = np.sum(np.logical_and(total_pred[:,0]==1,total_pred[:,0])).astype(float)
    
        precision =tp/(tp+fp)
        recall = tp/(tp+fn) 
        f1  = 2*(precision*recall)/(precision+recall)
        if f1> best_f1:
               best_f1 = f1
               best_epsilon = epsilon
    return best_epsilon,best_f1

def predict_anomalies(X,Ptrain,epsilon):
    
     anomalies = np.where(Ptrain<epsilon)
     plt.scatter(X[:,0],X[:,1],marker="2",s=70,alpha=0.4)
     plt.scatter(X[anomalies[0],0],X[anomalies[0],1],c='r',marker="2",s=100)
     plt.text(22.5,24,'ephsilon = 8.99E-05',horizontalalignment='right')
     plt.xlabel("Latency")
     plt.ylabel("Throughput")
     plt.show()
    
    
    


def main():
    data = sio.loadmat("C:/Users/europ/Desktop/ML_FOLDER/machine-learning-ex8/ex8/ex8data1.mat")
    Xtrain = data['X']
    Xval=data['Xval']
    Yval=data['yval']
    
    p =  np.zeros((Xtrain.shape[0],Xtrain.shape[1]))
    p_train =  np.zeros((Xval.shape[0],Xval.shape[1]))

    
    mu,sigma = estimate_gaussian_parameters(Xtrain)
    Ptrain = prob_distribution_train(p_train,Xtrain)
    Pval = prob_distribution_val(p,Xval,Yval)
    
    epsilon,f1 = select_threshold(Pval,Yval)
    predict_anomalies(Xtrain,Ptrain,epsilon)
    print(epsilon)
        
    
    
    
    
if __name__=='__main__':
    main()