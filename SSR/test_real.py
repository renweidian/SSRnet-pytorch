# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:37:27 2023

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
from model4 import *
import hdf5storage


    
dataset='CAVE'

net2 = CNN_BP_SE5()

net2.load_state_dict(torch.load('imx_REAL2.pkl'))
#net2=torch.load('imx_REAL2.pkl')
R=h5py.File('D:\paper-writing\SSRNet\srf.mat', 'r')
R=np.float32(np.array(R['b']))
R=R.transpose()
R_inv=np.linalg.pinv(R)
R_inv=torch.Tensor(R_inv)
R=torch.Tensor(R).cuda()




training_size=64
stride=32
net2.eval()

MSI=cv2.imread('RGB.bmp')

MSI=torch.Tensor(MSI)
MSI_1= torch.unsqueeze(MSI, 0)
MSI_1=torch.permute(MSI_1,[0,3,1,2])
print(MSI_1.shape)
with torch.no_grad():
     Fuse=reconstruction(net2,R,R_inv.cuda(),MSI_1.cuda(),training_size,stride)    
     Fuse=Fuse.cpu().detach().numpy()
Fuse=np.squeeze(Fuse)
faker_hyper = np.transpose(Fuse,(1,2,0))
print(faker_hyper.shape)
hdf5storage.savemat('real_hsi.mat', {'HSI': faker_hyper}, format='7.3')

