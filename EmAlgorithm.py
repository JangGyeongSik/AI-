import numpy as np

X = ['HTTTHHTHTH','HHHHTHHHHH','HTHHHHHTHH','HTHTTTHHTT','THHHTHHHTH']

def EM_step(ThetaA, ThetaB):
    Maximization_step = []

    for i in X:
        H = 0
        T = 0
        for j in i:
            if j == 'H':
                H += 1
            else: T += 1
        BernA = (ThetaA**H)*((1-ThetaA)**T)
        BernB = (ThetaB**H)*((1-ThetaB)**T)
        BernA_propor = BernA/(BernA+BernB)
        BernB_propor = BernB/(BernA+BernB)
        Maxi_Bern_propor = [BernA_propor*H, BernA_propor*T,BernB_propor*H,BernB_propor*T]
        #rint Maxi_BernA_propor, Maxi_BernB_propor
        Maximization_step.append(Maxi_Bern_propor)

    CoinAH, CoinAT, CoinBH, CoinBT = np.sum(np.array(Maximization_step), axis=0)
    Predict_theta_A = CoinAH / (CoinAH + CoinAT)
    Predict_theta_B = CoinBH / (CoinBH + CoinBT)

    return Predict_theta_A, Predict_theta_B

ThetaA = 0.6
ThetaB = 0.5

for i in range(5):
    ThetaA, ThetaB = EM_step(ThetaA, ThetaB)
    print (round(ThetaA,3), round(ThetaB,3))
