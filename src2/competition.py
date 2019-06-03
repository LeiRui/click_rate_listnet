feature_extraction(train_data,train_feature)
feature_extraction(test_data,test_feature)

metadata = {'scores':[0,10],'input_size' : 3}
train_data_loader = RankSet(train_feature,metadata)
test_data_loader = RankSet(test_feature,metadata)

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
model.train(train_data_loader)

outputs = []
with open('my_submission.csv', 'w') as f:
  for example in test_data_loader:
    outputs = model.use_learner(example)
    query_id = example[2]
    tot_title = example[3]
    for i in range(len(outputs)):
        f.write("{0},{1},{2}\n".format(query_id, tot_title[i], outputs[i][0]))