import numpy as np
import math

def make_datacard_2tag(outDataCardsDir,modelName,  signal_rate, normalization, bkg_rate, observation, bkg_unc, bkg_unc_name, sig_unc, sig_unc_name,signal_region, prefix):
    a,b,c,d = bkg_rate[0], bkg_rate[1], bkg_rate[2], bkg_rate[3]
    nSig = len(signal_rate.keys())
    text_file = open(outDataCardsDir+modelName+".txt", "w")
    text_file.write('# signal norm {0} \n'.format(normalization))

    text_file.write('imax {0} \n'.format(4))
    text_file.write('jmax {0} \n'.format(nSig))
    text_file.write('kmax * \n')
    text_file.write('shapes * * FAKE \n')


    text_file.write('--------------- \n')
    text_file.write('--------------- \n')
    text_file.write('bin \t chA \t chB \t chC \t chD \n')
    text_file.write('observation \t {0:6.2f} \t {1:6.2f} \t {2:6.2f} \t {3:6.2f} \n'.format(observation[0],observation[1],observation[2],observation[3]))
    text_file.write('------------------------------ \n')
    text_file.write('bin '+'\t chA ' * (1+nSig) + '\t chB ' * (1+nSig) +'\t chC '*(1+nSig) +'\t chD '*(1+nSig) +'\n')
    process_name = '\t '+ (' \t ').join(list(signal_rate.keys())) + '\t bkg '
    text_file.write('process ' + process_name * 4 + '\n')
    process_number = '\t '+ (' \t ').join(list((np.arange(nSig)*-1).astype(str))) + '\t 1'
    text_file.write('process ' + process_number * 4 + '\n')
    rate_string = 'rate'
    for i in range(4):# 4 bins
        for k,v in signal_rate.items():
            rate_string +='\t {0:e} '.format(v[i])
        rate_string += '\t 1 '
    text_file.write(rate_string+'\n')
    text_file.write('------------------------------ \n')

    text_file.write(prefix+'A   rateParam       chA     bkg      (@0*@2/@1)                    '+prefix+'B,'+prefix+'C,'+prefix+'D \n')
    if b == 0: text_file.write(prefix+'B   rateParam       chB     bkg     {0:.2f}        [0,{1:.2f}] \n'.format(b, c*7))
    else: text_file.write(prefix+'B   rateParam       chB     bkg     {0:.2f}        [0,{1:.2f}] \n'.format(b, b*7))
    text_file.write(prefix+'C   rateParam       chC     bkg     {0:.2f}        [0,{1:.2f}] \n'.format(c, c*7))
    if d == 0:text_file.write(prefix+'D   rateParam       chD     bkg     {0:.2f}        [0,{1:.2f}] \n'.format(d, c*7))
    else: text_file.write(prefix+'D   rateParam       chD     bkg     {0:.2f}        [0,{1:.2f}] \n'.format(d, d*7))


    for k,v in signal_rate.items():text_file.write('norm rateParam * {0} 1  \n'.format(k))

    #### signal uncertainties ####
    for k,v in sig_unc.items():assert(len(sig_unc_name)==len(v))
    for i in range(len(sig_unc_name)):
        unc_text = sig_unc_name[i]+' \t lnN'
        if len(sig_unc[list(sig_unc.keys())[0]][i])==4:#symmetric uncertainties
            for j in range(4):#bin
                for k,v in sig_unc.items():
                    if v[i][j] == 0.0:unc_text += ' \t -'
                    else: unc_text += ' \t '+str(v[i][j]+1)
                unc_text += '\t - '
        else:#asymmetric
            for j in range(4):#bin A, B, C, D
                for k,v in sig_unc.items():
                    if  v[i][j] == 0.0 and v[i][j+4] == 0.0: unc_text += ' \t -'
                    else:unc_text += ' \t {0}/{1}'.format(1-v[i][j],1+v[i][j+4])
                unc_text += '\t -'
        text_file.write(unc_text + ' \n')
            
            
    for i in range(len(bkg_unc_name)):
        bkg_unc_text = bkg_unc_name[i] + ' \t lnN ' + '\t - '*(4*nSig+3) + '\t ' + str(1+bkg_unc[i]) + ' \n'
        text_file.write(bkg_unc_text)
    

    text_file.close()
def readNorm(f_cscCard):
    f = open(f_cscCard,"r")
    norm = float(f.readline().split()[3])
    return norm


def HLT_CSC(eta,nstation,size):
    cond = (nstation == 1) & (np.abs(eta)<1.9) & (size >= 200)
    cond = cond | ((nstation == 1) & (np.abs(eta)>1.9) & (size >= 500))
    cond = cond | ((nstation > 1) & (np.abs(eta)<1.9) & (size > 100))
    cond = cond | ((nstation > 1) & (np.abs(eta)>1.9) & (size > 500))
    return cond

def L1_trg(cscClusterR, cscClusterZ, cscClusterSize):  
    first_in_ME11 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>100, cscClusterR<275), np.abs(cscClusterZ)>580), np.abs(cscClusterZ)<632) 
    first_in_ME12 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>275, cscClusterR<465), np.abs(cscClusterZ)>668), np.abs(cscClusterZ)<724)
    first_in_ME13 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>505, cscClusterR<700), np.abs(cscClusterZ)>668), np.abs(cscClusterZ)<724)
    first_in_ME21 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>139, cscClusterR<345), np.abs(cscClusterZ)>789), np.abs(cscClusterZ)<850)
    first_in_ME22 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>357, cscClusterR<700), np.abs(cscClusterZ)>791), np.abs(cscClusterZ)<850)
    first_in_ME31 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>160, cscClusterR<345), np.abs(cscClusterZ)>915), np.abs(cscClusterZ)<970)
    first_in_ME32 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>357, cscClusterR<700), np.abs(cscClusterZ)>911), np.abs(cscClusterZ)<970)
    first_in_ME41 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>178, cscClusterR<345), np.abs(cscClusterZ)>1002), np.abs(cscClusterZ)<1063)
    first_in_ME42 = np.logical_and(np.logical_and(np.logical_and(cscClusterR>357, cscClusterR<700), np.abs(cscClusterZ)>1002), np.abs(cscClusterZ)<1063)
    
    first_in_plateau_ME11 = np.logical_and(first_in_ME11, cscClusterSize>=500)
    first_in_plateau_ME21 = np.logical_and(first_in_ME21, cscClusterSize>=500)
    first_in_plateau_ME31 = np.logical_and(first_in_ME31, cscClusterSize>=500)
    first_in_plateau_ME41 = np.logical_and(first_in_ME41, cscClusterSize>=500)

    first_in_plateau_ME12 = np.logical_and(first_in_ME12, cscClusterSize>=200)
    first_in_plateau_ME13 = np.logical_and(first_in_ME13, cscClusterSize>=200)
    first_in_plateau_ME22 = np.logical_and(first_in_ME22, cscClusterSize>=200)
    first_in_plateau_ME32 = np.logical_and(first_in_ME32, cscClusterSize>=200)
    first_in_plateau_ME42 = np.logical_and(first_in_ME42, cscClusterSize>=200)
    
    first_in_plateau = first_in_plateau_ME11 | first_in_plateau_ME12 | first_in_plateau_ME13 | first_in_plateau_ME21 | first_in_plateau_ME22 | \
    first_in_plateau_ME31 | first_in_plateau_ME32 | first_in_plateau_ME41 | first_in_plateau_ME42
    return first_in_plateau

def deltaPhi( phi1,  phi2):
    dphi = phi1-phi2
    while np.count_nonzero(dphi > math.pi)>0:
        dphi[dphi > math.pi] -= 2*math.pi
    while np.count_nonzero(dphi< -math.pi)>0:
        dphi[dphi < -math.pi] += 2*math.pi
    return dphi
