import numpy as np
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn




class mergeToken(nn.Module):
    def __init__(self, dim):
        super(mergeToken, self).__init__()
        self.lin1 = nn.Linear(dim,dim,dtype=torch.bfloat16)
        self.lin2 = nn.Linear(dim,dim,dtype=torch.bfloat16)
        self.lin3 = nn.Linear(dim,dim,dtype=torch.bfloat16)
        self.layer_norm = nn.LayerNorm(dim,dtype=torch.bfloat16)
        self.activate = nn.ELU()
        
    def forward(self,input):
        output = input
        output = self.activate(self.lin1(output))
        output = self.activate(self.lin2(output))
        output = self.activate(self.lin3(output))
        return output

class simpleMergeModel(nn.Module):
    def __init__(self, n_dim, n_layers):
        super(simpleMergeModel, self).__init__()
        self.layers = nn.ModuleList(mergeToken(n_dim * 2) for _ in range(n_layers))
        self.lin1 = nn.Linear(n_dim * 2, 1, dtype=torch.bfloat16)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,input):
        #temp [layer_num, k_v, batch_size, num_heads, seq_len//2, 2*head_dim]
        temp = input.view(*input.size()[:-2], -1, 2 * input.size(5))
        for layer in self.layers:
            temp = layer(temp)
        #pow [layer_num, k_v, batch_size, num_heads, seq_len//2, head_dim]
        pow_1 = self.sigmoid(self.lin1(temp)).expand(-1, -1, -1, -1, -1, input.size(5))
        
        pow_2 = torch.ones_like(pow_1) - pow_1
        return input[:,:,:,:,0::2] * pow_1 + input[:,:,:,:,1::2] * pow_2 
        #return input[:,:,:,:,0::2] * 0.5 + input[:,:,:,:,1::2] * 0.5 


class mergeModel(nn.Module):
    def __init__(self, n_dim, n_layers):
        super(mergeModel, self).__init__()
        self.layers = nn.ModuleList(mergeToken(n_dim*2) for _ in range(n_layers))
        self.merge = nn.Linear(n_dim*2, n_dim, dtype=torch.bfloat16)
    def forward(self, input):
        """
        #print("size",input.size())
        output = input.view(input.size(0), input.size(1), input.size(2), input.size(3), -1, 2, input.size(5))
        output = output.mean(dim=5)
        #print("output",output.size())
        return output

        
        """ 
        
        output = input.view(input.size(0), input.size(1), input.size(2), input.size(3), -1, input.size(5)*2)
        for layer in self.layers:
            output = layer(output)
        output = nn.ELU()(self.merge(output)) 
        return output
        


class cacheManager():
    def __init__(self, args):
        super(cacheManager, self).__init__()
        self.past_kv = None
        self.mergePosition = []
        self.noRepeatNum = 0
        self.sum_token = 0
        self.length = args.merge.length
        self.count = 0
        self.generateOrder()
        if args.merge.path == None:
            self.model = simpleMergeModel(args.merge.n_dim, args.merge.n_layer).cuda()
        else:
            self.model = torch.load(args.merge.path).cuda()
            #self.cachemanager.model.from_pretrained(args.merge.path)

    def generateOrder(self):
        for _ in range(self.length * 2 - 1):
            self.mergePosition.append(-1)
        i = self.length
        while i >= 1 :
            for j in range(0, i-1):
                self.mergePosition.append(j)
                self.generateSubOrder(i*2)
            i//=2
        self.noRepeatNum = len(self.mergePosition)
        self.mergePosition.append(-2)
        self.generateSubOrder(2)


    def generateSubOrder(self, pos):
        if pos==self.length:
            self.mergePosition.append(pos-2)
            return
        elif pos>self.length:
            return
        self.generateSubOrder(pos*2)
        self.mergePosition.append(pos-2)
        self.generateSubOrder(pos*2)



    def inference(self):
        self.count += 1
        if self.count >= len(self.mergePosition):
            self.count = self.noRepeatNum
        temp = self.mergePosition[self.count] 
        if temp == -2:
            self.sum_token += 1
        elif temp != -1:
            position = temp + self.sum_token
            self.past_kv = torch.cat((self.past_kv[:,:,:,:,:position,:], 
                                      self.model(self.past_kv[:,:,:,:,position:position+2,:]), 
                                      self.past_kv[:,:,:,:,position+2:,:]), dim=4)

        
    def denseByGroup(self):
        cacheLength = self.past_kv[0][0].size(2)
        #print("cache",cacheLength)
        if cacheLength <= 2*self.length-2:
            self.past_kv = self.model(self.past_kv)
        else:
            self.sum_token += 1 
            self.past_kv = torch.cat((self.past_kv[:,:,:,:,:-2*self.length+2], 
                                      self.model(self.past_kv[:,:,:,:,:2*self.length-2,:])), 
                                      dim=4)
            
        
        
        
    def getInputs(self, chunk_len):
        if self.length == chunk_len:
            self.count += self.length 
            if self.count >= len(self.mergePosition):
                self.count = self.noRepeatNum + self.count - len(self.mergePosition)
            self.denseByGroup()
        else:
            for i in range(chunk_len):
                self.inference()
            

        
        """
        T = self.past_kv[0][0].size(2) // self.length
        for i in range(T):
            self.denseByGroup()
        for i in range(T * self.length, self.past_kv[0][0].size(0)):
            self.inference()
        """
        
            
    def train(self, model, inputs, train_func, optimizer, scheduler):
        
        total_loss = 0
        true_total_loss = 0
        T = inputs['input_ids'].size(-1) // self.length
        #print("lengeh",inputs['input_ids'].size(-1))
        if T * self.length > inputs['input_ids'].size(-1):
            T += 1
        for i in range(T):
            
            begin = i * self.length
            end = min((i + 1) * self.length, inputs['input_ids'].size(-1))
            #print(inputs['input_ids'].shape,begin,end)

            temp = {}
            for context in inputs:
                temp[context] = inputs[context][:,begin:end]
            loss = train_func(model, temp)

            true_total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i != T - 1:
                self.past_kv = self.past_kv.detach()
                self.denseByGroup()
                #pass
            total_loss += loss
        print("true_total_loss",true_total_loss)
        return total_loss
            
"""
        self.past_kv = torch.stack([torch.stack(row) for row in self.past_kv])
        self.past_kv = tuple( 
            tuple(self.past_kv[i, j] for j in range(self.past_kv.size(1))) 
            for i in range(self.past_kv.size(0))
        )

        
        """
               
"""


        self.pow = []
        self.limit = []
        self.generateLimit(length)

    def generateLimit(self, length):
        k=int(math.log2(length))
        for i in range(k):
            for j in range(int(math.pow(2,i))):
                self.limit.append(int(math.pow(2,k-i-1)))


    def inference(self):
        self.pow.append(1)
        if pow.length-1 > self.length:
            temp=0
            for j in range(self.length-1,-1,-1):
                if self.pow[j] < self.limit[j] and temp==0 or temp!=0 and self.pow[j] == self.pow[temp] and self.pow[j] < self.limit[j]:
                    temp=j
            if temp!=0 or self.pow[0] < self.length and temp==0:
                self.pow[temp] = self.pow[temp] * 2 
                del self.pow[temp+1]             
                self.mergeCache(temp)
            elif self.pow[0] >= self.length:
                self.sum_token += 1
                del self.pow[0]

    def mergeCache(self, temp):
        position = temp + self.sum_token
        batch_size = self.past_kv[0][0].size(0)
        size_1 = self.past_kv[0][0].size(1)
        size_3 = self.past_kv[0][0].size(3)
        for pask_kv_pre_layer in self.past_kv:
            for kv_in_past_kv in pask_kv_pre_layer:
                kv_in_past_kv = kv_in_past_kv.permute(2,0,1,3).contiguous().view(-1, batch_size, size_1 * size_3)
                kv_in_past_kv[position] = self.model(kv_in_past_kv[position], 
                                                    kv_in_past_kv[position+1])
                #del kv_in_past_kv[position+1]
                kv_in_past_kv = torch.cat((kv_in_past_kv[:position+1], kv_in_past_kv[position+2:]), dim=0)
                kv_in_past_kv = kv_in_past_kv.view(-1, batch_size, size_1, size_3).permute(1,2,0,3)
"""

    


