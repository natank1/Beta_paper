import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta

dx = 1000

if __name__ =='__main__':
    a1=100
    b1=100
    a2=150
    b2=70
    th_new = [i / dx for i in range(dx)]

    plt.plot(th_new, beta.pdf(th_new, 200,1), 'b')
    plt.plot(th_new, beta.pdf(th_new, 1,  200), 'k')

    plt.plot(th_new, beta.pdf(th_new, a1, b1), 'r')
    plt.plot(th_new, beta.pdf(th_new, a2, b2), 'g')
    plt.title("Narrow functions examples")
    plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.legend(["Left Narrow", "Right Narrow", "Fun1", "Func2"])

    plt.show()
