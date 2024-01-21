# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
from model4 import *

def reconstruction(net2,R,MSI,training_size,stride):
        index_matrix=np.zeros((R3.shape[1],MSI.shape[1],MSI.shape[2]))
        abundance_t=np.zeros((R3.shape[1],MSI.shape[1],MSI.shape[2]))
        for j in range(0, MSI.shape[1]-training_size+1, stride):
            for k in range(0, MSI.shape[2]-training_size+1, stride):
                temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                MSI_1=torch.Tensor(np.expand_dims(temp_hrms,axis=0))
                with torch.no_grad():
                  HSI = net2(R,MSI_1.cuda())
                HSI=HSI.cpu().detach().numpy()
                HSI=np.squeeze(HSI)
                HSI=np.clip(HSI,0,1)
                abundance_t[:,j:j+training_size, k:k+training_size]= abundance_t[:,j:j+training_size, k:k+training_size]+ HSI
                index_matrix[:,j:j+training_size, k:k+training_size]= 1+index_matrix[:,j:j+training_size, k:k+training_size]
                
        HSI_recon=abundance_t/index_matrix
        return HSI_recon
    
dataset='CAVE'
if dataset=='CAVE':
   path='D:\我的代码\高光谱集数据\CAVE\\'
elif dataset=='Harvard':
   path='D:\我的代码\高光谱集数据\CAVE\\'
imglist=os.listdir(path)
filters = np.load('D:/我的代码\Python-光谱超分/spectral super-resolution\
/NTIRE2020_spectral-master/resources/cie_1964_w_gain.npz')['filters']
filters=filters/255.0
filters=filters.T
R3=torch.Tensor(filters).cuda()
net2=CNN_BP_SE1(1e-5).cuda()
#net2.load_state_dict(torch.load('.\Weights\CAVE_60.pth'))
R=create_F()
R=torch.Tensor(R).cuda()




RMSE=[]
training_size=64
stride=32
for i in range(20,len(imglist)):
    net2.eval()
    img=loadmat(path+imglist[i])
    img1=img["b"]
    HRHSI=torch.Tensor(np.transpose(img1,(2,0,1)))
    MSI=torch.tensordot(R3.cpu(),  HRHSI, dims=([1], [0]))
    MSI_1= torch.unsqueeze(MSI, 0)
   
    with torch.no_grad():
         prediction = net2(R3,MSI_1.cuda())
         Fuse=prediction.cpu().detach().numpy()
#         Fuse=reconstruction(net2,R,MSI.cpu(),training_size,stride)    
        
    Fuse=np.squeeze(Fuse)
    Fuse=np.clip(Fuse,0,1)
    a,b=metrics.rmse1(Fuse,HRHSI)
    RMSE.append(a)  
print(np.mean(RMSE))
    
