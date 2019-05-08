
#===================ELMANFIS FOR 1 EPOCH


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import csv

#=============================LOAD TRAINING DATA==========================
filename = "../input/bell1154/Bell1154.csv"
raw_data = open(filename, 'rt')
trainData = np.loadtxt(raw_data, delimiter=",")
#==========================================================================


#==============================TRAINING====================================
nOutput=1
nInputs=np.size(trainData,1)-nOutput
minData=np.amin(trainData, axis=0)
maxData=np.amax(trainData, axis=0)

rangeInput=np.subtract((maxData),(minData))


nMembershipFn=4
ctr2ctrDist= np.true_divide(rangeInput,(nMembershipFn-1))

 #====================Estimating random mf parameters
c=np.zeros((nInputs,nMembershipFn))
b=np.zeros((nInputs,nMembershipFn))
a=np.zeros((nInputs,nMembershipFn))

for j in range(nInputs):
    for i in range(nMembershipFn):
        c[j,i]= ((random.random()-0.5)*ctr2ctrDist[j]) +(minData[j] + (i-1)*ctr2ctrDist[j])
        a[j,i]= (random.random()-0.5)*rangeInput[j]/(2*nMembershipFn-2)*2+rangeInput[j]/(2*nMembershipFn-2)
        b[j,i]= random.random()*0.2 + 1.9

        
X = np.array([])

#==========================Calculating membership grades
for m in range(trainData.shape[0]):
    
    membershipGrades=np.zeros((nInputs,nMembershipFn))
    for j in range(nInputs):
        for i in range(nMembershipFn):
            membershipGrades[j,i]=1/(1+(abs((trainData[m][j]-c[j,i])/a[j,i]))**(2*b[j,i]))
        
    #=================================Calculating firing strength
    B=np.zeros((nInputs,(nMembershipFn**nInputs)))
    for i in range(nInputs):
        t=0
        for k in range(int(nMembershipFn**(i))):
            for j in range(nMembershipFn):
                for l in range(nMembershipFn**(nInputs-i-1)):
                    B[i,t]=membershipGrades[i,j]
                    t=t+1
     
    weights=np.prod(B, axis = 0) 
    #==================================Calculating Normalised Firing
    weightNormalize=np.true_divide(weights,np.sum(weights))
    
    #==================================Generating X of f=XZ
    X1=weightNormalize.transpose()*trainData[m][0]
    for j in range(1,nInputs):
        X1=np.vstack([X1, trainData[m][j]*weightNormalize.transpose()])
        
    X1=np.vstack([X1,weightNormalize.transpose()])
   
   
    Xt=np.asarray(X1).ravel()
    
    if m==0:
        X=np.hstack((X,Xt.transpose()))
    else:
        X=np.vstack((X,Xt.transpose()))   

        
#==============Evaluating consequent parameters(Z) for first output by solving linear equation
Z1 = np.linalg.pinv(X) 
ZZ=np.matmul(Z1,trainData[:,nInputs])
        

#============================================TRAINING ENDS=======================================    


# In[ ]:

#====================================EVALUATE TRAINING ERROR============================
from sklearn.metrics import mean_squared_error
TRO=np.matmul(X,ZZ)
error=mean_squared_error(TRO,trainData[:,nInputs])
print(error)


# In[ ]:

#=============================LOAD TESTING DATA==========================
filenamewe = "../input/testbell/belltest.csv"
rawa_data = open(filenamewe, 'rt')
testData = np.loadtxt(rawa_data, delimiter=",")
#==================================================================

# In[ ]:

#===========================================TESTING============================
Xtest = np.array([])
for m in range(testData.shape[0]):
    
    membershipGradestest=np.zeros((nInputs,nMembershipFn))
    for j in range(nInputs):
        for i in range(nMembershipFn):
            membershipGradestest[j,i]=1/(1+(abs((testData[m][j]-c[j,i])/a[j,i]))**(2*b[j,i]))
        
    #Calculating firing strength
    Btest=np.zeros((nInputs,(nMembershipFn**nInputs)))
    for i in range(nInputs):
        t=0
        for k in range(int(nMembershipFn**(i))):
            for j in range(nMembershipFn):
                for l in range(nMembershipFn**(nInputs-i-1)):
                    Btest[i,t]=membershipGradestest[i,j]
                    t=t+1
     
    weightstest=np.prod(Btest, axis = 0) 
    #Calculating Normalised Firing
    weightNormalizetest=np.true_divide(weightstest,np.sum(weightstest))
    
    #Generating X of f=XZ
    X1t=weightNormalizetest.transpose()*testData[m][0]
    for j in range(1,nInputs):
        X1t=np.vstack([X1t, testData[m][j]*weightNormalizetest.transpose()])
        
    X1t=np.vstack([X1t,weightNormalizetest.transpose()])
   
   
    Xtt=np.asarray(X1).ravel()
    
    if m==0:
        Xtest=np.hstack((Xtest,Xtt.transpose()))
    else:
        Xtest=np.vstack((Xtest,Xtt.transpose()))   
#=========================================================================================================        


#TEST OUTPUT================================================
TTO=np.matmul(Xtest,ZZ)


# In[ ]:

#=====================TEST  ERROR==========================================
errortest=mean_squared_error(TTO,testData[:,nInputs])
print(errortest)


# showing important matrices=======================================
print(a)
print(b)
print(c)
print(X)
print(Xtest)
print(ZZ)