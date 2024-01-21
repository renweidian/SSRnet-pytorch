# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 00:12:12 2020

@author: Dian
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:20:42 2020

@author: Dian
"""

"""
Created on Sat May 23 22:06:42 2020

@author: Dian
"""
#-*- coding:utf-8 -*-
from model3 import *
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

R=create_F()
stride=32
training_size=64
LR=2e-4
EPOCH=100
BATCH_SIZE=64
init_lr2=5e-4
init_lr1=init_lr2/10
rmse_optimal=6
decay_power=1.5
num=20
maxiteration=math.ceil(((512-training_size)//stride+1)**2*num/BATCH_SIZE)*EPOCH
#maxiteration=math.ceil(((1040-training_size)//stride+1)*((1392-training_size)//stride+1)*num/BATCH_SIZE)*EPOCH
warm_iter=math.floor(maxiteration/40)
print(maxiteration)
dataset='CAVE'
if dataset=='CAVE':
   path1='D:\\code\\cave\\cave_train\\'
   path2='D:\\code\\cave\\caveall\\'
elif dataset=='Harvard':
    path1='D:\\codeall\\code\\CZ_hsdb_indoor\\harved_train\\'
    path2='D:\\codeall\\code\\CZ_hsdb_indoor\\harved_test\\'
imglist1=os.listdir(path1)
imglist2=os.listdir(path2)
train_data=HSI_MSI_Data1(path1,R,training_size,stride,num=20)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
R_inv=np.linalg.pinv(R)

R_inv=torch.Tensor(R_inv)
R2=torch.Tensor(R)
R2=R2.cuda()
cnn=CNN_BP_SE5().cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
loss_func=LossTrainCSS()
loss_func=nn.L1Loss().cuda()

for m in cnn.modules():
  if isinstance(m, (nn.Conv2d, nn.Linear)):
    nn.init.xavier_uniform_(m.weight) 

step=0
for epoch in range(EPOCH):
    print(epoch)
    time1 = time.time()
    cnn.train()
    for step1, (a1, a2) in enumerate(train_loader):
        lr=warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,step, lr_decay_iter=1,  max_iter=maxiteration, power=decay_power)
        step=step+1
        output = cnn(R2,a2.cuda())
        loss=loss_func(output, a1.cuda())
        optimizer.zero_grad()          
        loss.backward()                
        optimizer.step()

    cnn.eval()
    if epoch%5==0:
        RMSE=[]
        SAM=[]
        for i in range(0,len(imglist2)):
            cnn.eval()
            img=loadmat(path2+imglist2[i])
            img1=img["b"]
            #img1=img1/img1.max()
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            MSI_1=torch.Tensor(np.expand_dims(MSI,axis=0))

            
            with torch.no_grad():
#                  prediction = cnn(R2,R_inv.cuda(),MSI_1.cuda())
              prediction=reconstruction(cnn,R2,R_inv.cuda(),MSI_1.cuda(),training_size,stride)
              Fuse=prediction.cpu().detach().numpy()
              
              
            Fuse=np.squeeze(Fuse)
            Fuse=np.clip(Fuse,0,1)
            a,b=metrics.rmse1(Fuse,HRHSI)
            c=metrics.sam(Fuse,HRHSI)*180/pi
            RMSE.append(a)
            SAM.append(c)
#            if np.mean(RMSE)<rmse_optimal:
        rmse_optimal=np.mean(RMSE)

        time2 = time.time()
        run_time = time2-time1
        print(epoch,lr,np.mean(RMSE),np.mean(SAM))
        print("time is:%.9f" %run_time)
        torch.save(cnn, 'cave/' +str(epoch)+ 'last.pkl')
