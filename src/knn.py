from src.methods import *
class KNN:

  def __init__(self, k=5, metric='euclidean',target='classification', dev=None):
    #Parameter fail safes
    if not isinstance(k, int) or k <0:
      raise ValueError('k must be a non-negative integer')
    possible_metrics={'euclidean', 'manhattan','mahalanobis', 'cosine similarity'}
    if not (isinstance(metric, str) and metric in {'euclidean', 'manhattan', 'mahalanobis', 'cosine similarity'} or callable(metric)):
      raise ValueError('Invalid metric. Allowed metrics are \'euclidean\' , \'manhattan\',\'mahalanobis\'or \'cosine similarity\'; or a callable function')
    possible_targets={'classification', 'regression'}
    if target not in possible_targets:
      raise ValueError('Invalid target. Allowed metrics are \'classification\' or \'regression\'')
    if (not isinstance(dev, int) or dev <0) and (dev is not None):
      raise ValueError('Standard deviation must be a non-negative integer')



    self.k=k #parameter amount of neighbors
    self.metric=metric #parameter type of metric
    self.target=target #parameter target (classification or regression)
    try: #parameter deviation for accepted deviation in regression models, if it isnt regression this parameter doesnt exist
      if target=='classification' and dev!=0 and (dev is not None):
        self.dev=None
        raise ValueError("Deviation won't be used in classification models, parameter ignored")
      else:
        self.dev=dev
    except ValueError as e:
      print(e)
    self.cov_inv=0

  def fit(self, X, y): #fit in KNN stores the data to compare the given point
    self.X_train=X
    self.y_train=y
    if self.metric=='mahalanobis':
      self.cov_inv=np.linalg.inv(np.cov(X, rowvar=False))
    #check that klen(X):
      raise ValueError("k is bigger than the amount of data points, recreate KNN with a different k, or add more data")



  def predict(self, X): # calls make prediction method for each point of X
    y_pred=[self.make_prediction(x) for x in X]
    return np.array(y_pred)


  def make_prediction(self, x):
    #calculates the distance depending on the metric chosen
    distances=[calculate_distance(x, x_train, self.metric, self.cov_inv) for x_train in self.X_train]
    #sort distance, returns indexes of the closest k
    nearest_index=np.argsort(distances)[:self.k]
    #supervised learning compares with true value, get value of closest
    nearest_y_value=[self.y_train[i] for i in nearest_index]
    if self.target=='regression' and self.dev==None:
      self.dev=np.std(nearest_y_value)
    #get y value depending on the target
    return find_target(nearest_y_value, self.target)

  def evaluate(self, y_pred, y_test, eval='accuracy', custom_dev=None):
    if self.target == 'classification':
      try:
        if eval != 'accuracy':
            raise ValueError("eval parameter is ignored in classification models, evaluation method is accuracy")
      except ValueError as e:
        print(e)
      try:
        if custom_dev is not None:
          raise ValueError("Deviation won't be used in classification models, parameter ignored")
      except ValueError as e:
        print(e)
      return class_accuracy(y_pred, y_test)
    elif self.target == 'regression':
        if eval == 'accuracy':
            if custom_dev is not None:
              if (not isinstance(custom_dev, int) or custom_dev <0) and (custom_dev is not None):
                raise ValueError('Standard deviation must be a non-negative integer')
              elif custom_dev >= (max(y_test) - min(y_test)):
                raise ValueError("Deviation higher than y range, guaranteed 100% accuracy, no significance in results")
              self.dev=custom_dev
            elif self.dev is None:
                raise ValueError("For regression evaluation with 'accuracy', you must specify a non-negative deviation (dev).")
            elif self.dev >= (max(y_test) - min(y_test)):
                raise ValueError("Deviation higher than y range, guaranteed 100% accuracy, no significance in results")
            else:
              self.dev = self.dev
            return reg_accuracy(y_pred, y_test, self.dev)
        elif eval == 'score':
            return mean_squared_error(y_pred, y_test)
        else:
            raise ValueError("Invalid eval. Allowed evaluation methods are: 'accuracy' or 'score'")
    else:
        raise ValueError("Invalid target. Allowed targets are 'classification' or 'regression'")



  def cross_validation(self, X, y, folds=5, eval='accuracy'): #cross validation for how well the model does splitting the data in different ways
    shuffle_index=np.random.permutation(len(X))
    fold_size=len(X)//folds
    fold_index=[shuffle_index[i:i+fold_size] for i in range(0, len(X), fold_size)]
    metrics=[]
    for i in range(folds):
      test_indices = fold_index[i]
      train_indices = np.concatenate([fold_index[j] for j in range(folds) if j != i])
      X_train = X[train_indices]
      y_train = y[train_indices]
      X_test = X[test_indices]
      y_test = y[test_indices]
      self.fit(X_train, y_train)
      y_pred = self.predict(X_test)
      metrics.append(self.evaluate(y_pred, y_test, eval))
    return np.mean(metrics)
