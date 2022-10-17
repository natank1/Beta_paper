import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import beta
from scipy.special import psi, gamma,digamma
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from training.beta_package import kldiff,kl2
dx = 1000
nepoch = 7
thr_new =   [i / dx for i in range(dx)]
thr_min =0.00001
def calc_beta(th_new, prb0, labels,dx=1000):
    predt_custom =[1 if labels[i] ==1   else  0 for i in range(len(labels))]
    neg_prb = [prb0[i] for i in range(len(prb0)) if predt_custom[i] == 0]
    pos_prb = [prb0[i] for i in range(len(prb0)) if predt_custom[i] == 1]

    lr_f11 = [len([j for j in pos_prb if (j < th_new[i])]) / len(pos_prb) for i in range(dx)]
    tr_f11 = [len([j for j in neg_prb if (j < th_new[i])]) / len(neg_prb) for i in range(dx)]
    a, b = getparams(np.mean(tr_f11), np.power(np.std(tr_f11), 2))
    al, bl = getparams(np.mean(lr_f11), np.power(np.std(lr_f11), 2))
    print("tt  ", a, b,np.mean(tr_f11),np.power(np.std(tr_f11), 2))
    print("ll ", al, bl,np.mean(lr_f11), np.power(np.std(lr_f11), 2))
    return a,b,al,bl


def mysoftprob_obj(predt: np.ndarray, data: xgb.DMatrix):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    '''
    labels = data.get_label()
    kRows=len(labels)
    kClasses =2
    weights = np.ones((kRows, 1), dtype=float)
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)
    extra_grad =np.zeros((kRows, kClasses), dtype=float)
    eps = 1e-6
    special_prob=predt.shape[0]*[0]
    for r in range(predt.shape[0]):
        target = labels[r]

        p = softmax(predt[r, :])
        m0= min(p)
        m1 = max(p)
        bool0 =m1>m0
        special_prob[r] =p[kClasses-1]
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            # g=g+extra_grad[r,c]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    min_var = thr_min
    mn_mean = 0.1 * thr_min
    at, bt, al, bl = calc_beta(thr_new, special_prob, labels)
    # bt= np.max(bt,min_var)
    # bl = np.max(bl, min_var)

    if bt <= min_var:
        super_grad_1 = 0
    else:
        super_grad_1 = kldiff(at, bt, 1, min_var)

    if bl <= min_var:
        super_grad_2 = 0
    else:
        super_grad_2 = kldiff(al, bl, mn_mean, min_var)

    for r in range(predt.shape[0]):
        if labels[r] ==1:
            grad[r,1] +=0.01*super_grad_1
        else:
            grad[r, 0] += 0.01*super_grad_2
    print ("ssss ",super_grad_1,super_grad_2)
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess

def softmax(x):
    '''Softmax function with x as input vector.'''
    e = np.exp(x)
    return e / np.sum(e)

def get_params(labels,probs,th_new):
    neg_prb = [probs[i] for i in range(len(probs)) if labels[i] == 0]
    pos_prb = [probs[i] for i in range(len(probs)) if labels[i] == 1]
    lr_f11 = [len([j for j in pos_prb if (j < th_new[i])]) / len(pos_prb) for i in range(dx)]
    tr_f11 = [len([j for j in neg_prb if (j < th_new[i])]) / len(neg_prb) for i in range(dx)]
    a, b = getparams(np.mean(tr_f11), np.power(np.std(tr_f11), 2))
    al, bl = getparams(np.mean(lr_f11), np.power(np.std(lr_f11), 2))
    return a,b, al,bl
def calc_confusion(y_true,y_p,n_class):

    mm = confusion_matrix(y_p, y_true)
    print(mm)
    index_acc= n_class=1
    print ("Accuracy=",accuracy_score(y_p, y_true))
    print ("Precison :",  mm[index_acc,index_acc]/np.sum(mm[:,index_acc]))
    print (mm[:,index_acc])

    return

def beta_kl(a1, b1, a2, b2):
  """https://en.wikipedia.org/wiki/Beta_distribution"""
  B = beta
  DG =digamma
  return -np.log(gamma(a2+ b2))    + np.log(gamma(a2))+np.log(gamma(b2)) -np.log(gamma(a1)) - np.log(gamma(b1))  + np.log(gamma(a1+ b1)) +(a1 - a2) *(  DG(a1)-DG(a1 + b1))  + (b1 - b2) * (DG(b1)-DG(a1 + b1))

def getparams(mu, var):
  mu2 =mu*mu
  alpha = (  ((1 - mu)/var) - (1 / mu)) * mu2
  beta = alpha * (1 / mu - 1)
  return alpha,   beta
def train_proccs():
    X, y = make_classification(n_samples=10000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    print (X_train.shape,X_test.shape,y_test.shape,y_train.shape)
    th_new = [i / dx for i in range(dx)]
    xgtrain = xgb.DMatrix(X_train, label=y_train)



    custom_results = {}
    mm = xgb.train({'num_class': 2,
                                'disable_default_eval_metric': True},
                               xgtrain,
                               num_boost_round=nepoch,
                               obj=mysoftprob_obj,
                               # custom_metric=merror,
                               evals_result=custom_results,
                               evals=[(xgtrain, 'train')])

    xgtest = xgb.DMatrix(X_test, label=y_test)
    labels =xgtest.get_label()
    pred0= mm.predict(xgtest)
    pred1= [1 if i>0.5 else 0 for i in pred0]
    accuracy = accuracy_score(y_test, pred1)

    print("accuracy=", accuracy)
    print ("MCC=" ,matthews_corrcoef(labels, pred1))
    calc_confusion(labels, pred1, 2)
    a,b,al,bl = get_params(labels,pred0,th_new)
    print("KL param=",a,b,al,bl )
    print("KL ", beta_kl(a, b, al, bl))
    x = np.linspace(beta.ppf(0.0001, a, b), beta.ppf(0.9999, a, b), 100)
    plt.figure(figsize=(7, 7))
    plt.xlim(0, 1.0001)

    plt.plot(th_new, beta.pdf(th_new, a, b), 'r')
    plt.plot(th_new, beta.pdf(th_new, al, bl), 'g')

    plt.title('Beta Distribution '+str(nepoch), fontsize='15')
    plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.show()


if __name__ =='__main__':
    train_proccs()
