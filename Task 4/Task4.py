#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastNLP.io import Conll2003NERPipe
from fastNLP.embeddings import get_embeddings, BertEmbedding
from fastNLP.models import BiLSTMCRF
from fastNLP import Trainer, LossInForward, SpanFPreRecMetric
import torch
device = 0 if torch.cuda.is_available() else 'cpu'
data_paths={'train': './data/train.conll', 'dev': './data/dev.conll', 'test': './data/test.conll'}
data_bundle = Conll2003NERPipe().process_from_file(data_paths)
data_bundle.rename_field('chars', 'words')  # 这是由于BiLSTMCRF模型的forward函数接受的words，而不是chars，所以需要把这一列重新命名
vocab = data_bundle.get_vocab('words')
embed = BertEmbedding(vocab=vocab, model_dir_or_name='en', auto_truncate=True)
model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200, dropout=0.5,
              target_vocab=data_bundle.get_vocab('target'))
optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-5)
loss = LossInForward()
metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
trainer = Trainer(train_data=data_bundle.get_dataset('train'), dev_data=data_bundle.get_dataset('dev'),
    batch_size=32, model=model, loss=loss, optimizer=optimizer, metrics=metric, device=device )
trainer.train()
tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
tester.test()


# In[6]:


torch.cuda.empty_cache()


# ## Reference
# 
# https://github.com/fastnlp/fastNLP/blob/master/docs/source/_static/notebooks/tutorial_4_load_dataset.ipynb
# https://blog.csdn.net/weixin_43909659/article/details/120210053
# https://github.com/fastnlp/fastNLP/blob/master/docs/source/tutorials/%E5%BA%8F%E5%88%97%E6%A0%87%E6%B3%A8.rst
