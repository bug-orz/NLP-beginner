#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tqdm
from torchtext import data
import transformers
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm.notebook import tqdm
import pickle
import pandas as pd

epochs=30
num_classes = 2350
batch_size=32
max_length=15
train_path='./data/train.csv'
test_path='./data/test.csv'
bert_name='bert-base-chinese'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class BertDataset(Dataset):
    def __init__(self, tokenizer,max_length,data_path):
        super(BertDataset, self).__init__()
        self.train_csv=pd.read_csv(data_path)
        self.tokenizer=tokenizer
        self.target=self.train_csv.iloc[:,1]
        self.max_length=max_length
        
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, index):
        
        text1 = self.train_csv.iloc[index,0]
        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.long),
            'length': torch.tensor(len(text1), dtype=torch.long)
            }

tokenizer = transformers.BertTokenizer.from_pretrained(bert_name)
train_dataset= BertDataset(tokenizer, max_length,train_path)
test_dataset= BertDataset(tokenizer, max_length,test_path)
train_loader=DataLoader(train_dataset,batch_size=batch_size)
dev_loader=DataLoader(test_dataset,batch_size=batch_size)


# In[3]:


# 定义LSTM模型

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, output_dim,  
                 bidirectional, dropout):
        super(LSTM,self).__init__()     
        self.name='LSTM'
        self.bert = transformers.BertModel.from_pretrained(bert_name)
        #self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(TEXT.vocab.vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.ReLU()

    def forward(self,ids,mask,token_type_ids,text_length):
        embedded,_= self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length, batch_first=True,enforce_sorted=False)
        #print(embedded)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)

        return outputs
    
embed_size = 768
hidden_size = 256
num_layers = 1
bidirectional = True
dropout_rate = 0.1

model = LSTM(embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate)


# In[4]:


for param in model.bert.parameters():
    param.requires_grad = False
loss_fn = nn.CrossEntropyLoss()
#Initialize Optimizer
optimizer= torch.optim.Adam(model.parameters())

def train(epochs,train_loader,dev_loader,model,loss_fn,optimizer):
    model.train()
    for  epoch in range(epochs):
        #print(epoch)
        for batch, dl in tqdm(enumerate(train_loader),leave=False,total=len(train_loader)):
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']
            text_length=dl['length']
            optimizer.zero_grad()
            
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
                text_length=text_length)

            loss=loss_fn(output,label)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            corr_num = 0
            err_num = 0
            for batch, dl in tqdm(enumerate(dev_loader),leave=False,total=len(dev_loader)):
                ids=dl['ids']
                token_type_ids=dl['token_type_ids']
                mask= dl['mask']
                label=dl['target']
                text_length=dl['length']
                optimizer.zero_grad()

                outputs=model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids,
                    text_length=text_length)

                corr_num += (outputs.argmax(1) == label).sum().item()
                err_num += (outputs.argmax(1) != label).sum().item()
            tqdm.write('Epoch {}, Accuracy {}'.format(epoch, corr_num / (corr_num + err_num))) 
        torch.save(model, './model/model_'+model.name+'_epoch_{}.pkl'.format(epoch))
            
    return model

model=train(epochs, train_loader,dev_loader, model, loss_fn, optimizer)


# In[5]:


with open("./data/vocab_index", "rb") as fp:   # Unpickling   
    vocab = pickle.load(fp)
model=torch.load('./model/model_LSTM_epoch_0.pkl')
model.eval()


# In[6]:


def encode_text(text):
    inputs = tokenizer.encode_plus(
            text ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=15
        )
    ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    mask = inputs["attention_mask"]
    ids=torch.tensor(ids, dtype=torch.long).view(1,len(ids)),
    mask=torch.tensor(mask, dtype=torch.long).view(1,len(mask)),
    token_type_ids=torch.tensor(token_type_ids, dtype=torch.long).view(1,len(token_type_ids)),
    #length=torch.tensor(len(text), dtype=torch.long).view(1,len(text))
    length=torch.LongTensor(max_length+10).cpu()#.view(1,len(text))
    print(ids)
    model.eval()
    output=model(
                ids=ids[0],
                mask=mask[0],
                token_type_ids=token_type_ids[0],
                text_length=length)
    return vocab[output.argmax(1)]

encode_text('是')


# In[ ]:


torch.tensor(len('是多少'), dtype=torch.long)


# In[10]:


def gen_char_helper(dev_loader,model):
    model.eval()
    with torch.no_grad():
        corr_num = 0
        err_num = 0
        for batch, dl in (enumerate(dev_loader)):
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']
            text_length=dl['length']
            optimizer.zero_grad()

            outputs=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
                text_length=text_length)
            return vocab[outputs.argmax(1)[0]]

def gen_char(text):
    pd.DataFrame({'text':[text],'label':[1]}).to_csv('./data/sample.csv',index=False)
    sample_dataset= BertDataset(tokenizer, max_length,'./data/sample.csv')
    sample_loader=DataLoader(sample_dataset,batch_size=1)
    return gen_char_helper(sample_loader,model)

def gen_para(text):
    ans=text
    temp=''
    for i in range(5):
        temp=gen_char(ans)
        ans+=temp
    return ans[:]

gen_para('汪')


# In[ ]:




