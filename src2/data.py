import csv
def feature_extraction(input_path, output_path):
  with open(output_path, 'w') as f:
    with open(input_path,'r',encoding='UTF-8') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        query_id = int(row[0])
        query = row[1]
        title_id = int(row[2])
        title = row[3]
        if(len(row) == 5):
            label = int(row[4])
        else:
            label = -1 # test set NOTE
            
        if(label == 1):#TODO
          label = 10  #TODO

        # extract feature from (query,title) pair
        hit_cnt = 0
        query_w = query.split()
        for w in title.split():
          if(w in query_w):
            hit_cnt += 1
        # TODO 再添加query中词的平均词频、title中不属于query中词的平均词频
        

        # input,target,query
        f.write("{0} {1} {2},{3},{4},{5}\n".format(len(query),len(title),hit_cnt,label,query_id,title_id))

def sample_div(input_path, output1_path, output2_path):
  with open(input_path, "r") as input, open(output1_path, "w") as output1, open(output2_path, "w") as output2:
    row_num = 0
    for row in input:
      if row_num < 17000:
        output1.write("{0}".format(row))
      else:
        output2.write("{0}".format(row))
      row_num += 1

class RankSet(object):
  def __init__(self,path=None, metadata={}): # TODO batch_size
    self.path = path
    self.metadata = metadata

  def __iter__(self):
    tot_input = []
    tot_target = []
    tot_title = []
    last_query = None

    with open(self.path,'r',encoding='UTF-8') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        input_feature = row[0].split()
        input_feature[0] = int(input_feature[0])
        input_feature[1] = int(input_feature[1])
        input_feature[2] = int(input_feature[2])
        # TODO 以后有的feature是float

        target = int(row[1])
        query_id = int(row[2])
        title_id = int(row[3])

        if(last_query != None):
          if(last_query != query_id):
            yield (tot_input,tot_target,last_query,tot_title)
            tot_input = []
            tot_target = []
            tot_title = []

        tot_input.extend([input_feature])
        tot_target.extend([target])
        tot_title.extend([title_id])
        last_query = query_id

      if tot_input: # Output last ranking example
        yield (tot_input,tot_target,last_query, tot_title)

# if __name__ == '__main__':
#   raw_data = "../data/train_data.sample"
#   feature_data = "../data/feature_data.sample"
#   feature_data_train = "../data/feature_data_train.sample"
#   feature_data_test = "../data/feature_data_test.sample"
#
#   feature_extraction(raw_data,feature_data)
#
#   sample_div(feature_data,feature_data_train,feature_data_test)
#
#   metadata = {'scores':[0,10],'input_size' : 3}
#   train_data_loader = RankSet(feature_data_train,metadata)
#   test_data_loader = RankSet(feature_data_test,metadata)
#
#   cnt = 0
#   for input, target, query in train_data_loader:
#     if(cnt < 10):
#       print(input,target,query)
#     cnt += 1
#
#   print("")
#   cnt = 0
#   for input, target, query in test_data_loader:
#     if(cnt < 10):
#       print(input,target,query)
#     cnt += 1