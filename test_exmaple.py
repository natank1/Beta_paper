import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import beta
from scipy.special import psi, gamma,digamma
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

dx = 1000
nepoch = 12


def get_params(labels,probs,th_new):
    neg_prb = [probs[i] for i in range(len(probs)) if labels[i] == 0]
    pos_prb = [probs[i] for i in range(len(probs)) if labels[i] == 1]
    lr_f11 = [len([j for j in pos_prb if (j < th_new[i])]) / len(pos_prb) for i in range(dx)]
    tr_f11 = [len([j for j in neg_prb if (j < th_new[i])]) / len(neg_prb) for i in range(dx)]
    a, b = getparams(np.mean(tr_f11), np.power(np.std(tr_f11), 2))
    al, bl = getparams(np.mean(lr_f11), np.power(np.std(lr_f11), 2))
    return a,b, al,bl

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
def test_proccs():
    X, y = make_classification(n_samples=10000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    print (X_train.shape,X_test.shape,y_test.shape,y_train.shape)
    print (max(y),min(y))
    th_new = [i / dx for i in range(dx)]
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    params_constr={
        'base_score':0.5,
        'learning_rate':0.1,
        'max_depth':5,
        'min_child_weight':100,
        'n_estimators':200,
        'nthread':-1,
         'objective':'binary:logistic',
        'seed':2018,
        'eval_metric':'auc'
    }


    mm= xgb.train(params_constr,xgtrain,num_boost_round=nepoch)
    xgtest = xgb.DMatrix(X_test, label=y_test)
    labels =xgtest.get_label()
    pred0= mm.predict(xgtest)
    pred1= [1 if i>0.5 else 0 for i in pred0]
    accuracy = accuracy_score(y_test, pred1)

    print("accuracy=", accuracy)
    print ("MCC=" ,matthews_corrcoef(labels, pred1))
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
    test_proccs()
