from __future__ import division

import sys
import csv
import math
import simple_ml
import time
import pdb
from heapq import nsmallest
from pprint import pprint
from collections import defaultdict
    
clean_instances = simple_ml.load_instances('agaricus-lepiota.data', filter_missing_values=False)
attribute_names = simple_ml.load_attribute_names('agaricus-lepiota.attributes')
attribute_names_and_values = simple_ml.load_attribute_names_and_values('agaricus-lepiota.attributes')

def isfloat(x):
  try:
      a = float(x)
  except ValueError:
      return False
  else:
      return True

def isint(x):
  try:
      a = float(x)
      b = int(a)
  except ValueError:
      return False
  else:
      return a == b

def cleanfloats(instances):
  """Rounds all floats
  Args:
  instances -- list of instances.
  """
  for instance in instances:
    for attb_index in range(0,len(instance)):
      if isfloat(instance[attb_index]):
        if(instance[attb_index].isdigit()):
          instance[attb_index] = int(instance[attb_index])
        else:
          instance[attb_index] = round(float(instance[attb_index]), 0)
  return instances

if len(sys.argv) < 3:
  sys.exit(('Usage: {} trainingdata.csv testdata.csv').format(sys.argv[0]))
else:
  with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f)
    trainingdata = list(reader)[1:]
    trainingdata = cleanfloats(trainingdata)
  with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    testdata = list(reader)[1:]
    testdata = cleanfloats(testdata)

def entropy(instances, classifier, class_index):
  """Calculate the entropy across instances of a binary classified dataset.
  Args:
  instances -- array of data instances
  classifier -- character or number that represents a positive classification (i.e. 1, y, etc.)
  class_index -- index in an instance which contains the classification (often last)
  """

  value_counts = defaultdict(int)
  class_counts = defaultdict(int)

  for instance in instances:
    index = instance[class_index]
    if index!='?':
      if isint(index):
        index = int(index)
      value_counts[index] += 1

  # pdb.set_trace()

  num_values = len(value_counts)
  if num_values <= 1:
      return 0

  entropies = []
  for key, value in value_counts.iteritems():
    if key==classifier:
      pos_class = value
    else:
      neg_class = value

  instance_count = len(instances) # counter for # of instances

  try:
    pos_entropy = -((pos_class/instance_count)*math.log(pos_class/instance_count, 2))
    neg_entropy =  -((neg_class/instance_count)*math.log(neg_class/instance_count, 2))
  except ValueError, e:
    print ("No good hombre. Log error. \n" + 
            "pos_class= {}, neg_class = {}, instance_count = {}").format(pos_class, neg_class, instance_count)
    # print instances
    raise e
  return pos_entropy + neg_entropy

def information_gain(instances, classifier, split_index, class_index):
  """Returns a sorted list of attributes ranked by information gain
  Args:
  instances -- array of data instances
  classifier -- character or number that represents a positive classification (i.e. 1, y, etc.)
  split_index -- index on which the data is split
  class_index -- index in an instance which contains the classification (often last)
  """

  set_entropy = entropy(instances, classifier, class_index)
  subset_instances = defaultdict(list)
  for instance in instances:
      subset_instances[instance[split_index]].append(instance)
  subset_entropy = 0.0
  num_instances = len(instances)
  for subset in subset_instances:
      subset_prob = len(subset_instances[subset])/num_instances
      subset_entropy += subset_prob * entropy(
        subset_instances[subset], classifier, class_index)
  return set_entropy - subset_entropy

def split_instances(instances, attb_index):
  """Splits instances based on attribute at attb_index. Returns dict with format:
  {
    <attb_value>: [[instance with that attb_value for attb], [another instance]],
    <attb_value2>: [[etc.]]
  }

  Args:
  instances -- array of data instances
  attb_index -- the index of the attribute to split the instances on
  """

  split_instances = defaultdict(list)
  for instance in instances:
    pdb.set_trace()
    attb = instance[attb_index]
    split_instances[attb].append(instance)
  return split_instances

def choose_best_attb_index(instances, classifier, class_index, index_list):
  """Chooses the best attribute to split on based on information gain. Returns its index.
  Args:
  instances -- array of instances
  classifier -- character or number that represents positive classification
  class_index -- index of the classification in instance
  index_list -- list of possible attribute indices
  """

  max_info = 0
  max_info_index = 0
  for i in range(1,len(index_list)):
    info = information_gain(instances, classifier, i, class_index)
    if info>max_info:
      max_info = info
      max_info_index = i
  return max_info_index

def common_value(instances, classifier, class_index):
  """Returns the most common value for an attribute across instances.
  Args:
  instances -- array of instances
  classifier -- character or number that represents positive classification
  class_index -- index of classifier.
  """
  values = defaultdict(int)
  for instance in instances:
    values[instance[class_index]]+=1

  maxcount = 0

  for key, value in values.iteritems():
    if value>maxcount:
      maxcount = value
      common_value = key
  return common_value

def make_tree(instances, classifier, class_index, depth = 0):
    '''Returns a decision tree made from instances
    Args:
    instances -- array of instances
    classifier -- character or number that represents positive classification
    class_index -- index of classifier.
    '''
  
    attribute_range = range(len(instances[0]))
    attribute_range.remove(class_index) # don't want to split on the classifier

    pos_neg_counts = defaultdict(int)
    for instance in instances:
      pos_neg_counts[instance[class_index]]+=1
    
    if len(pos_neg_counts) == 1 or (len(pos_neg_counts)==2 and '?' in pos_neg_counts.keys()): # only positive or only negative
      class_label = common_value(instances, classifier, class_index)
      return class_label        
    else:
        best_index = choose_best_attb_index(instances, classifier, class_index, range(1, len(attribute_range)))        

        tree = {best_index:{}}
        leaves = split_instances(instances, best_index)

        for attribute_value in leaves:
          subtree = make_tree(leaves[attribute_value],
                              classifier,
                              class_index,
                              depth+1)

          tree[best_index][attribute_value] = subtree

    return tree

def get_majority(tree, classifier, class_index):
  counter = 0
  for key, value in tree.iteritems():
    if value == classifier:
      counter+=1
    else:
      counter-=1
  return counter > 0

def get_close_neighbor_value(tree, classifier, class_index, attribute):
  closestkeys = nsmallest(10, tree.keys(), key=lambda x: 100 if (isinstance(attribute, basestring) or isinstance(x, basestring)) else abs(x-attribute))
  counter = 0
  for key in closestkeys:  
    if not isinstance(tree[key], dict):
      if tree[key] == classifier:
        counter+=1
      else:
        counter-=1
  return 1 if counter>0 else 0




def classify(tree, instance, classifier, class_index):
  """Returns the classification of an instance.
  Args:
  tree -- decision tree for classifying
  instance -- a single instance to classify
  classifier -- caharacter or number that represents positive classification
  """

  # pdb.set_trace()

  if not isinstance(tree, dict): # means tree is no longer a dict and just the classificaton
      # print "Right here"
      return tree

  attb_index = tree.keys()[0]
  subtree = tree.values()[0]

  attribute = instance[attb_index]

  if attribute in subtree:
    return classify(subtree[attribute], instance, classifier, class_index)
  else:
    if attribute=='?':
      return get_majority(subtree, classifier, class_index)
    else:
      return get_close_neighbor_value(subtree, classifier, class_index, attribute)

trainingdata_slice = trainingdata[1:int(len(trainingdata)//2)]
testdata_slice = trainingdata[int(len(trainingdata)//2+1):]

tree = make_tree(trainingdata_slice, 1, len(trainingdata[0])-1)
pprint(tree)
# pdb.set_trace()
correct_count = 0
for instance in testdata_slice:
    predicted_label = classify(tree, instance, 1, len(trainingdata[0])-1)
    actual_label = instance[len(trainingdata[0])-1]
    if predicted_label==actual_label:
      correct_count+=1

print 'Accuracy is {} out of {}, {} percent'.format(correct_count, len(testdata_slice), correct_count/len(testdata_slice))
    # print 'predicted: {}; actual: {}'.format(predicted_label, actual_label)
