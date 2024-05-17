#Other methods that don't need to belong to the class
import pandas as pd
import numpy as np

def euclidean_distance(x1,x2): #for euclidean distance
  return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1,x2): #for manhattan distance
  return np.sum(np.abs(x1-x2))

def mahalanobis_distance(x1,x2,cov_inv): #mahalanobis distance
  #cov_inv = np.linalg.inv(np.cov(X_train, rowvar=False)) passed directly from fit so its not calculated each time
  return np.sqrt(np.dot(np.dot((x1-x2), cov_inv), (x1-x2)))

def cosine_similarity(x1,x2):
  return np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

def most_common(values): #for classification target
  counts={}
  for item in values:
    counts[item]=counts.get(item, 0)+1
  sorted_counts=sorted(counts.items(),key=lambda x: x[1], reverse=True)
  return sorted_counts[0][0]

def calculate_distance(x1,x2,metric,cov_inv):
  if metric=='euclidean':
    return euclidean_distance(x1,x2)
  if metric=='manhattan':
    return manhattan_distance(x1,x2)
  if metric=='mahalanobis':
    return mahalanobis_distance(x1,x2,cov_inv)
  if metric=='cosine similarity':
    return cosine_similarity(x1,x2)
  if callable(metric):
    try:
      return metric(x1, x2)
    except ValueError as e:
      print('Error in callable function: ',e)


def find_target(nearest_y_value,target):
  if target=='classification':
    return most_common(nearest_y_value)
  if target=='regression':
    return np.mean(nearest_y_value)

def class_accuracy(y_pred, y_test):
    error=0
    for i in range(len(y_pred)):
      if y_pred[i]!=y_test[i]:
        error=error+1
    return (len(y_pred)-error)/len(y_pred)*100

def reg_accuracy(y_pred, y_test, dev):
  error=0
  for i in range(len(y_pred)):
    if abs(y_pred[i]-y_test[i])>dev:
      error=error+1
  return (len(y_pred)-error)/len(y_pred)*100

def mean_squared_error(y_pred, y_test):
  squared_errors = (y_test - y_pred) ** 2
  mse = np.mean(squared_errors)
  return mse
