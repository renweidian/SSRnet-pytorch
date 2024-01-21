# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
from check1.model3 import *
import hdf5storage


    
dataset='CAVE'

path=''

imglist=os.listdir(path)
net2=torch.load('')
R=create_F()
R_inv=np.linalg.pinv(R)
R_inv=torch.Tensor(R_inv)
R=torch.Tensor(R).cuda()
test_path = ''
  



RMSE=[]
training_size=64
stride=32
for i in range(0,len(imglist)):
    net2.eval()
    img=loadmat(path+imglist[i])
    img1=img["ref"]
    img1=img1/img1.max() 
#    print("real_hyper's shape =",img1.shape)
    HRHSI=torch.Tensor(np.transpose(img1,(2,0,1)))
    MSI=torch.tensordot(R.cpu(),  HRHSI, dims=([1], [0]))
    MSI_1= torch.unsqueeze(MSI, 0)
     
    with torch.no_grad():
#         Fuse = net2(R,R_inv.cuda(),MSI_1.cuda())
         Fuse=reconstruction(net2,R,R_inv.cuda(),MSI_1.cuda(),training_size,stride)    
         Fuse=Fuse.cpu().detach().numpy()
    Fuse=np.squeeze(Fuse)
    Fuse=np.clip(Fuse,0,1)
    faker_hyper = np.transpose(Fuse,(1,2,0))
    print(faker_hyper.shape)
    a,b=metrics.rmse1(Fuse,HRHSI)
    RMSE.append(a) 
    test_data_path=os.path.join(test_path+imglist[i])
    hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
    hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')
print(np.mean(RMSE))
    
