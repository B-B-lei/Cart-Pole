import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F


class CSVLoader(Dataset):
    def __init__(self,csv_path,state_dim=8,action_dim=2):
        df=pd.read_csv(csv_path)
        start=2
        states_np=df.iloc[:,start:start+state_dim].values.astype(np.float32)
        actions_np=df.iloc[:,start+state_dim:start+state_dim+action_dim].values.astype(np.float32)

        #input需要标准化,后续C中写forward也要对theta/dot 进行标准化,参数写进去;
        self.mean=np.mean(states_np,axis=0)
        self.std=np.std(states_np,axis=0)
        #actions需要归一化(与Tahn对齐),除max,后续需要还原(和电机控制逻辑匹配)
        self.scale=np.max(np.abs(actions_np))
        #print("states_np, self.mean's type is:"type(states_np),type(self.mean))
        self.states=torch.tensor((states_np-self.mean)/ (self.std+1e-8) )
        self.actions=torch.tensor(actions_np/self.scale)

        self.num_samples=len(df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 严格共用同一个随机行号 idx，保证跨矩阵的因果绝对对齐
        return self.states[idx], self.actions[idx]            


class BaseModel(nn.Module):
    def __init__(self,state_dim=8,action_dim=2,hidden_dim=64):
        super().__init__()
        self.actor_net=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,32),
            nn.Tanh(),
            nn.Linear(32,action_dim),
            nn.Tanh()
        )
    def forward(self,obs):
        return self.actor_net(obs)
        

if __name__=="__main__":
    data=CSVLoader(csv_path="simudata/lqr_expert_data.csv", state_dim=8, action_dim=2)
    data_loader = DataLoader(dataset=data, 
                             batch_size=64, 
                             shuffle=True, 
                             drop_last=True,
                             num_workers=4)
    model=BaseModel(state_dim=8,action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss=float("inf")
    for epoch in range(5):
        total_loss=0
        for batch_idx,(state,Target_action) in enumerate(data_loader):
            action=model(state)
            loss=F.mse_loss(action,Target_action)
            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 ==0:
                print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")    
        # 训练循环内部
        epoch_loss=total_loss/len(data_loader)
        if epoch_loss < best_loss:
            best_loss=epoch_loss
            torch.save(model.state_dict(), "actor_net.pth") # 👈 存盘