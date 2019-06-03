import numpy as np
from sklearn.metrics import roc_auc_score
import csv

def default_merge(input, query):
  return input

def err_and_ndcg(output,target,max_score,k=10):
  """
  Computes the ERR and NDCG score
  (taken from here: http://learningtorankchallenge.yahoo.com/evaluate.py.txt)
  """

  err = 0.
  ndcg = 0.
  l = [int(x) for x in target]
  r = [int(x)+1 for x in output]
  nd = len(target) # Number of documents
  assert len(output)==nd, 'Expected %d ranks, but got %d.'%(nd,len(r))

  gains = [-1]*nd # The first element is the gain of the first document in the predicted ranking
  assert max(r)<=nd, 'Ranks larger than number of documents (%d).'%(nd)
  for j in range(nd):
    gains[r[j]-1] = (2.**l[j]-1.0)/(2.**max_score)
  assert min(gains)>=0, 'Not all ranks present.'

  p = 1.0
  for j in range(nd):
    r = gains[j]
    err += p*r/(j+1.0)
    p *= 1-r

  dcg = sum([g/np.log(j+2) for (j,g) in enumerate(gains[:k])])
  gains.sort()
  gains = gains[::-1]
  ideal_dcg = sum([g/np.log(j+2) for (j,g) in enumerate(gains[:k])])
  if ideal_dcg:
    ndcg += dcg / ideal_dcg
  else:
    ndcg += 0.5

  return (err,ndcg)

class MyListNet(object):

  def __init__(self, n_stages, hidden_size = 50,
      learning_rate = 0.01,
      weight_per_query = False,
      alpha = 1.,
      merge_document_and_query = default_merge,
      seed = 1234):

    self.n_stages = n_stages
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.weight_per_query = weight_per_query
    self.alpha = alpha
    self.merge_document_and_query = merge_document_and_query
    self.seed = seed

    self.stage = 0

  def initialize_learner(self,metadata):
    self.rng = np.random.mtrand.RandomState(self.seed)
    input_size = metadata['input_size']
    self.max_score = max(metadata['scores'])
    self.V = (2*self.rng.rand(input_size,self.hidden_size)-1)/input_size

    self.c = np.zeros((self.hidden_size))
    self.W = (2*self.rng.rand(self.hidden_size,1)-1)/self.hidden_size
    self.b = np.zeros((1))


  def update_learner(self,example):
    input_list = np.array(example[0])
    relevances = example[1]
    query = example[2]
    n_documents = len(input_list)

    target_probs = np.zeros((n_documents,1))
    input_size = input_list[0].shape[0]
    inputs = np.zeros((n_documents,input_size))

    for t,r,il,input in zip(target_probs,relevances,input_list,inputs): #逐document
      t[0] = np.exp(self.alpha*r)
      input[:input_size] = self.merge_document_and_query(il,query)
    target_probs = target_probs/np.sum(target_probs,axis=0)

    hid = np.tanh(np.dot(inputs,self.V)+self.c)

    outact = np.dot(hid,self.W) + self.b
    outact -= np.max(outact)
    expout = np.exp(outact)
    output = expout/np.sum(expout,axis=0)

    doutput = output-target_probs
    dhid = np.dot(doutput,self.W.T)*(1-hid**2)

    if self.weight_per_query:
      lr = self.learning_rate*n_documents
    else:
      lr = self.learning_rate
    self.W -= lr * np.dot(hid.T,doutput)
    self.b -= lr * np.sum(doutput)
    self.V -= lr * np.dot(inputs.T,dhid)
    self.c -= lr * np.sum(dhid,axis=0)

  def use_learner(self,example): #这么用就可以list wise了
    input_list = np.array(example[0])
    n_documents = len(input_list)
    query = example[2]

    input_size = input_list[0].shape[0]
    inputs = np.zeros((n_documents,input_size))
    for il,input in zip(input_list,inputs):
      input[:input_size] = self.merge_document_and_query(il,query)

    hid = np.tanh(np.dot(inputs,self.V)+self.c)
    outact = np.dot(hid,self.W) + self.b
    outact -= np.max(outact)
    expout = np.exp(outact)
    output = expout/np.sum(expout,axis=0)

    # ordered = np.argsort(-output.ravel())
    # order = np.zeros(len(ordered))
    # order[ordered] = range(len(ordered))
    # return order
    return output

  # def cost(self,output,example):
  #   return err_and_ndcg(output,example[1],self.max_score)

  def train(self,trainset):
    if self.stage == 0:
      self.initialize_learner(trainset.metadata)
    for it in range(self.stage,self.n_stages):
      for example in trainset:
        self.update_learner(example)
    self.stage = self.n_stages


  def my_train(self,trainset,valset):
    if self.stage == 0:
      self.initialize_learner(trainset.metadata)
    for it in range(self.stage,self.n_stages):
      print("stage",it)
      for example in trainset:
        self.update_learner(example) #TODO batch
        
      outputs,costs = self.test(valset)
      filter_costs = []
      for c in costs:
          if(c==None):
              continue
          filter_costs.append(c)
      print(np.mean(filter_costs))
    
    self.stage = self.n_stages
    
    
  def use(self,dataset):
    outputs = []
    for example in dataset:
      outputs += [self.use_learner(example)]
    return outputs


  def test(self,dataset):
    outputs = self.use(dataset)
    costs = []
    for example,output in zip(dataset,outputs):
      # costs += [self.cost(output,example)]
      tmp = sum(example[1])
      if(tmp != 0 and tmp != len(example[1])*self.max_score): 
        costs += [roc_auc_score(example[1],output)]
      else:
        costs += [None] #all 0 or 1 cannot count auc
    return outputs,costs