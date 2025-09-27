import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MAPE(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(torch.exp(gap)-1)
        return torch.sum(mape)/batch_size_ 

class prod(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        pro=-torch.mul(output,target)
        return torch.sum(pro)/batch_size_ 

class adj_mse(nn.Module):
    def __init__(self,mean_fact):
        super(adj_mse, self).__init__()
        self.mean_fact=mean_fact
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mse=torch.sum(torch.pow(gap,2))/batch_size_
        sbias=torch.pow(torch.sum(gap)/batch_size_,2)
        
        return mse + sbias*self.mean_fact

class over_bound(nn.Module):
    def __init__(self,bound,bound_fact):
        super(over_bound, self).__init__()
        self.bound=bound
        self.bound_fact=bound_fact
    def forward(self,output):
        batch_size_ = output.size()[0] # 获得batch_size
        over_bound=torch.abs(self.bound-output)+torch.abs(output+self.bound)-2*self.bound
        over_bound=torch.sum(torch.pow(over_bound,2))/batch_size_
        
        return over_bound*self.bound_fact
    
class over_bound_oneside(nn.Module):
    def __init__(self,bound,bound_fact):
        super(over_bound_oneside, self).__init__()
        self.bound=bound
        self.bound_fact=bound_fact
    def forward(self,output):
        batch_size_ = output.size()[0] # 获得batch_size
        over_bound=torch.abs(output-self.bound)+output-self.bound
        over_bound/=2
        over_bound=torch.sum(torch.pow(over_bound,2))/batch_size_
        
        return over_bound*self.bound_fact


class mean_value(nn.Module):
    def __init__(self,fact):
        super(mean_value, self).__init__()
        self.fact=fact
    def forward(self,output):
        batch_size_ = output.size()[0]
        mean_v=torch.abs(torch.sum(output)/batch_size_)
        
        return mean_v*self.fact

class mean_std_value(nn.Module):
    def __init__(self,mean_fact,std_fact):
        super(mean_std_value, self).__init__()
        self.mean_fact= mean_fact
        self.std_fact = std_fact
    def forward(self,output):
        batch_size_ = output.size()[0]
        mean_v=torch.abs(torch.sum(output)/batch_size_)
        std_v=torch.pow(torch.sum(torch.pow(output,2))/batch_size_, 0.5)
        
        return mean_v*self.mean_fact+std_v*self.std_fact

class penalty_L12(nn.Module):
    def __init__(self,fact1,fact2):
        super(penalty_L12, self).__init__()
        self.fact1=fact1
        self.fact2=fact2
    def forward(self,output):
        batch_size_ = output.size()[0] # 获得batch_size
        mape = self.fact1*torch.abs(output)+torch.pow(output,2)*self.fact2
        return torch.sum(mape)/batch_size_




class exp_penalty(nn.Module):
    def __init__(self,fact):
        super(exp_penalty, self).__init__()
        self.fact=fact
    def forward(self,output):
        batch_size_ = output.size()[0] # 获得batch_size
        out=torch.exp(torch.pow(output,2))
        over_bound=torch.sum(out)/batch_size_
        
        return over_bound*self.fact

class MAPE_taylor(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(gap+torch.pow(gap,2)/2+
                       torch.pow(gap,3)/6)
        return torch.sum(mape)/batch_size_
   
class MAPE_taylor_square(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(torch.pow(gap,2)+
                       torch.pow(gap,3))
        return torch.sum(mape)/batch_size_

    
class MAPE_taylor_sym(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(gap)+torch.pow(gap,2)/2
        return torch.sum(mape)/batch_size_

class mix_loss_L12(nn.Module):
    def __init__(self,fact1,fact2):
        super(mix_loss_L12, self).__init__()
        self.fact1=fact1
        self.fact2=fact2
    def forward(self,output,target,indic):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape1=torch.abs(gap)+torch.pow(gap,2)/2/self.fact1
        mape2=torch.abs(gap)+torch.pow(gap,2)/2/self.fact2
        mape = mape1*indic+(~indic)*mape2
        return torch.sum(mape)/batch_size_


class loss_L12(nn.Module):
    def __init__(self,fact):
        super(loss_L12, self).__init__()
        self.fact=fact
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(gap)+torch.pow(gap,2)/2*self.fact
        return torch.sum(mape)/batch_size_



class MAPE_taylor_sym_3(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(gap)+torch.pow(gap,2)/2+torch.abs(torch.pow(gap,3)/6)
        return torch.sum(mape)/batch_size_

class MAPE_pool(nn.Module):
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=torch.abs(torch.exp(output)-torch.exp(target))
        expy=torch.exp(target)
        return gap.sum()/expy.sum()+torch.abs(output.sum())/batch_size_+torch.abs(torch.pow(output,2)/batch_size_-1)
class weighted_l1(nn.Module):
    def forward(self,output,target):
        gap=torch.abs(output-target)*target
        weight = torch.sum(target)
        return torch.sum(gap)/weight
    
