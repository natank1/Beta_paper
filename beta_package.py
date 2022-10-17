#https://math.stackexchange.com/questions/257821/kullback-liebler-divergence#comment564291_257821

import numpy as np
from scipy.special import psi, gamma,beta,digamma,polygamma
print (polygamma(0,0.33),psi(0.33),digamma(0.33))

thr_min =0.00001
class beta_pack:
    def __init__(self,alpha,beta):
        self.alpha =alpha
        self. beta = beta
        if self.beta <thr_min:
          self.mu =self.alpha
          self.std_of_beta =0.0
        else:
            self.mu  = self.alpha/(self.alpha+self.beta)
            tt =self.alpha+self.beta
            self.std_of_beta = np.sqrt (  (self.alpha*self.beta)/ (tt*tt*(tt+1)))
        return
    def sum_alpha_beta(self):
        return self.alpha+self.beta

    def dalphadm(self,mu, sigma):
        if self.beta <thr_min :
            return [0.]

        return  ([2*mu-3*mu*mu] / (sigma*sigma)) - 1.

    def dalphadsig(self,mu, sigma):
        if self.beta<thr_min:
            return [0]
        return      (-2*mu*(1-mu)) / (sigma*sigma*sigma)


    def dbetadm(self, mu, sigma):
        if self.beta<thr_min:
            return [0]

        return (1/mu - 1) *self.dalphadm(mu,sigma)-(self.alpha / (mu*mu))


    def dbetasig(self, mu, sigma):
        if self.beta <thr_min:
            return [0.]

        return ((1 /mu) - 1) * self.dalphadsig(mu,sigma)


def kl(a1, b1, a2, b2):
  """https://en.wikipedia.org/wiki/Beta_distribution"""
  B = beta
  DG =digamma
  return np.log(B(a2, b2) / B(a1, b1)) + (a1 - a2) * DG(a1) + (b1 - b2) * DG(b1) + (
        a2 - a1 + b2 - b1) * DG(a1 + b1)
def kl2(a1, b1, a2, b2):
  """https://en.wikipedia.org/wiki/Beta_distribution"""
  B = beta
  DG =digamma

  return -np.log(gamma(a2+ b2))    + np.log(gamma(a2))+np.log(gamma(b2)) -np.log(gamma(a1)) - np.log(gamma(b1))  + np.log(gamma(a1+ b1)) +(a1 - a2) *(  DG(a1)-DG(a1 + b1))  + (b1 - b2) * (DG(b1)-DG(a1 + b1))

def derive_kl_product_term(delta0,deriv_detla0,a1,deriva,b1,derivsum):
    if (a1<thr_min) or b1<thr_min:
          return 0

    delta_dig =cover_digamma(a1)-cover_digamma(a1+b1)
    deriv_dig =polygamma(1,a1)*deriva -polygamma(1,a1+b1)*derivsum

    return deriv_detla0*delta_dig +delta0*deriv_dig

def cover_digamma(x):
    if x<thr_min:
        return 0
    return digamma(x)
def kldiff(a1, b1, a2, b2):
    beta1 =beta_pack(a1,b1)
    beta2 = beta_pack(a2, b2)
    beta2_dalphadm= beta2.dalphadm(beta2.mu,beta2.std_of_beta)[0]
    beta2_dbetadm = beta2.dbetadm(beta2.mu, beta2.std_of_beta)[0]
    beta2_dalphadm =0
    beta2_dbetadm=0
    deriv_mu_sum_2 = beta2_dalphadm+beta2_dbetadm
    beta1_dalphadm= beta1.dalphadm(beta1.mu,beta1.std_of_beta)[0]
    beta1_dbetadm = beta1.dbetadm(beta1.mu, beta1.std_of_beta)[0]
    deriv_mu_sum_1 = beta1_dalphadm + beta1_dbetadm
    delta_a = beta1.alpha -beta2.alpha
    delta_b = beta1.beta - beta2.beta

    delta_a_deriv =  beta1_dalphadm- beta2_dalphadm
    delta_b_deriv = beta1_dbetadm- beta2_dbetadm

    totlal_deriv= -cover_digamma(a2+b2)*deriv_mu_sum_2 +cover_digamma(a2)*beta2_dalphadm +cover_digamma(b2) *beta2_dbetadm  +  cover_digamma(a1+b1)*(deriv_mu_sum_1)-\
                     cover_digamma(a1) * beta1_dalphadm - cover_digamma(b1) * beta1_dbetadm  + derive_kl_product_term(delta_a,delta_a_deriv,beta1.alpha,beta1_dalphadm,beta1.beta,deriv_mu_sum_1)\
                     + derive_kl_product_term(delta_b,delta_b_deriv,beta1.beta,beta1_dbetadm,beta1.alpha,deriv_mu_sum_1)
    return totlal_deriv

if __name__ =='__main__':
    a=0
    print (kl(1, 2.5, 1, 3))
    print(kl2(1, 2.5, 1, 3))
    mm= kldiff(2, 1, 3, 4)
    print (mm)