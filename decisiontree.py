from __future__ import division

import sys
import os
import csv
import math
import simple_ml
import time
import pdb
import traceback
import itertools
import threading
import time
from heapq import nsmallest
from pprint import pprint
from collections import defaultdict
from copy import deepcopy



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

  # pdb.set_trace()

  max_info = 0
  max_info_index = 0
  for i in range(0,len(index_list)):
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
  
    attribute_range = range(0,len(instances[0]))
    attribute_range.remove(class_index) # don't want to split on the classifier

    pos_neg_counts = defaultdict(int)
    for instance in instances:
      pos_neg_counts[instance[class_index]]+=1
    
    if (len(pos_neg_counts)==2 and '?' in pos_neg_counts.keys()): # only positive or only negative
      class_label = [key for key in pos_neg_counts.keys() if key!='?']
      return class_label
    elif len(pos_neg_counts) == 1:
      return pos_neg_counts.keys()[0]
    else:
      max_tuple = max(pos_neg_counts.items(), key=lambda a: a[1])
      if float(max_tuple[1])/len(instances) > .79:
        return max_tuple[0]
      else:
        best_index = choose_best_attb_index(instances, classifier, class_index, range(0, len(attribute_range)))        
        # pdb.set_trace()
        tree = {best_index:{}}
        leaves = split_instances(instances, best_index)

        # group leaves together

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

  try:
    attb_index = tree.keys()[0]
    subtree = tree.values()[0]
  except IndexError, e:
    print('classifying issue')
    pdb.set_trace()
    raise e

  try:
    attribute = instance[attb_index]
  except Exception, e:
    print('attb issue')
    pdb.set_trace()
    raise e

  if attribute in subtree:
    if isinstance(subtree[attribute], dict) and not len((subtree[attribute])):
      return get_majority(subtree, classifier, class_index)
    return classify(subtree[attribute], instance, classifier, class_index)
  else:
    if attribute=='?':
      return get_majority(subtree, classifier, class_index)
    else:
      return get_close_neighbor_value(subtree, classifier, class_index, attribute)

def prune(whole_tree, path, subtree, validationdata):
  sys.stdout.write(' '*40)
  sys.stdout.flush()
  sys.stdout.write('pruning tree at path: {}\r'.format('-'.join(str(x) for x in path)))
  sys.stdout.flush()
  if isinstance(subtree, dict):
    if not len(subtree):
      path.pop()
    for leaf in subtree.items():
      # pdb.set_trace()
      path.append(leaf[0])
      if check_prune(whole_tree, path, validationdata):
        whole_tree = clip_leaf(whole_tree, path)
        path.pop()
      else:
        prune(whole_tree, path, leaf[1], validationdata)
    path.pop()
  elif isinstance(subtree, int):
    if check_prune(whole_tree, path, validationdata):
      whole_tree = clip_leaf(whole_tree, path)
      path.pop()
    # pdb.set_trace()
    path.pop()
    return
  else: # it's a fucking '?'
    path.pop()
    return

def popper(d,sequence):
  if len(sequence)<=1:
    return {}
  else:
    d[sequence[0]] = popper(d.pop(sequence[0]), sequence[1:])
    return d

def check_prune(tree, path, validationdata):
  tree_copy = deepcopy(tree)
  tree_copy = popper(tree_copy, path) # gets us to the subtree
  # pdb.set_trace()
  if validate(tree_copy, validationdata) > validate(tree, validationdata):
    sys.stdout.write('\npruning, increase from {} to {} \n'.format(validate(tree, validationdata), validate(tree_copy, validationdata)))
    sys.stdout.flush()
    return True
  else:
    return False

def clip_leaf(whole_tree, path):
  return popper(whole_tree, path)

def validate(tree, validationdata):
  correct_count=0

  for instance in validationdata:
    try:
      predicted_label = classify(tree, instance, 1, len(trainingdata[0])-1)
    except IndexError, e:
      print('you hit an index issue in classify. check out tree.keys()')
      pdb.set_trace()
      pprint(tree)
      raise e
    actual_label = instance[len(trainingdata[0])-1]
    if predicted_label==actual_label:
      correct_count+=1

  # print '{} classified correct, {} percent'.format(correct_count, correct_count/len(validationdata))
  return correct_count/len(validationdata)

def boolprint(tree):
  # pdb.set_trace()
  attributes = ['Winning percentage of team 1', 'Winning percentage of team 2', 'weather', 'the temperature', '# of injured on team 1', '# of injured on team 2', 'ERA of pitcher 1',
                'ERA of pitcher 2', 'days since team 1\'s last game', 'days since team 2\'s last game', 'at team 1\s home', 'run differential of team 1', 'run differential of team 1', 'winner']
  # pdb.set_trace()
  if isinstance(tree, dict):
    # pdb.set_trace()
    if len(tree.keys())==1:
      children = tree[tree.keys()[0]]
      for childkey in children.keys():
        sub = boolprint(children[childkey])
        return 'if {} is {}, {}'.format(attributes[tree.keys()[0]], childkey, sub)
  elif isinstance(tree,int):
    return 'then {}'.format(tree)
  elif isinstance(tree,str):
    return 'then {}'.format(tree)

if len(sys.argv) < 4:
  sys.exit(('Usage: {} trainingdata.csv validationdata.csv testdata.csv -prune -print').format(sys.argv[0]))
else:
  with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f)
    trainingdata = list(reader)[1:]
    trainingdata = cleanfloats(trainingdata)
  with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    validationdata = list(reader)[1:]
    validationdata = cleanfloats(validationdata)
  with open(sys.argv[3], 'r') as f:
    reader = csv.reader(f)
    testdata = list(reader)[1:]
    testdata = cleanfloats(testdata)

  tree = make_tree(trainingdata, 1, len(trainingdata[0])-1, 1)
  # pdb.set_trace()

  print 'Initial tree is at {}% accuracy.'.format(validate(tree, validationdata)*100)
  if len(sys.argv) > 4:
    if '-prune' in sys.argv:
      print('---Pruning tree--- \n')
      prune(tree, [tree.keys()[0]], tree[tree.keys()[0]], validationdata)

    if '-print' in sys.argv:
      print('---Printing tree--- \n')
      print boolprint(tree)
