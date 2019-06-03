from data import *
from model import *

raw_data = "../data/train_data.sample"
feature_data = "../data/feature_data.sample"
feature_data_train = "../data/feature_data_train.sample"
feature_data_test = "../data/feature_data_test.sample"

feature_extraction(raw_data,feature_data)
sample_div(feature_data,feature_data_train,feature_data_test)

metadata = {'scores':[0,10],'input_size' : 3}
train_data_loader = RankSet(feature_data_train,metadata)
test_data_loader = RankSet(feature_data_test,metadata)

# view
cnt = 0
for input, target, query, titles in train_data_loader:
  if(cnt < 10):
    print(input,target,query)
  cnt += 1
print("")
cnt = 0
for input, target, query, titles in test_data_loader:
  if(cnt < 10):
    print(input,target,query)
  cnt += 1

# train
model = MyListNet(1)
model.my_train(train_data_loader,test_data_loader)

outputs = model.use(test_data_loader)
print(outputs[:5])

outputs = []
with open('my_submission.csv', 'w') as f:
  for example in test_data_loader:
    outputs = model.use_learner(example)
    query_id = example[2]
    tot_title = example[3]
    for i in range(len(outputs)):
        f.write("{0},{1},{2}\n".format(query_id, tot_title[i], outputs[i][0]))