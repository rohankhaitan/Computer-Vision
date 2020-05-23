import cv2
import numpy as np
import pickle
from PA4_utils import load_image, load_image_gray
#import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.cluster.vq import vq
#from sklearn.cluster import MiniBatchKMeans


def build_vocabulary(image_paths, vocab_size):
  
  ## Using sift from opencv
  ## set no of keypoints 
  sift = cv2.xfeatures2d.SIFT_create(nfeatures = 200)
  for path in image_paths:

        img = cv2.imread(path)
        
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ## Compute the descriptors
        (_, descriptors) = sift.detectAndCompute(gray, None)
        
        ## Compute the final array of descriptors
        try:
                X = np.concatenate((X,descriptors))
        except:
                X = descriptors
  

  #cluster_centers
  kmeans = KMeans(n_clusters= vocab_size, random_state=0).fit(X)
  vocab = kmeans.cluster_centers_


  return vocab


def get_bags_of_sifts(image_paths, vocab_filename):
  
  ## Using sift from opencv
  ## set no of keypoints
  sift = cv2.xfeatures2d.SIFT_create(nfeatures = 300)

  with open(vocab_filename, 'rb') as f:
    vocab= pickle.load(f)

  ## Histogram of counts from the assignments
  def histogram(assignments):
    
    hist =[0 for i in range(vocab.shape[0])]

    for i in assignments:
        hist[i]+= 1
    
    hist = [i/sum(hist) for i in hist]
   
    return hist
    
  ## TF-IDF 

  def tfidf(tf_hists):
    
    tfidf = []
    df = []
    T = len(tf_hists)

    for i in range(vocab.shape[0]):
      temp = 0
      for hist in tf_hists:
        if hist[i] > 0:
          temp+= 1
      df.append(temp)
    idf = np.log([T/i for i in df])

    for hist in tf_hists:
      tfidf.append(hist*idf)

    return np.array(tfidf)  

  feats = []
  tf_hists = []

  for path in image_paths:

        img = cv2.imread(path)
        
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ## Compute the sift descriptors
        (_, descriptors) = sift.detectAndCompute(gray, None)
        assignments,_= vq(descriptors, vocab)
        
        tf_hists.append(histogram(assignments))

        # feats.append(histogram(assignments))
        
        
  
  # if TF_IDF ==True:
  #   feats = tfidf(tf_hists)
  #   return feats
  # else:
  #   feats = np.array(tf_hists)
  #   return feats
  
  feats = tfidf(tf_hists)
  #feats = np.array(tf_hists)
  return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
   
    ## Define the model
    neigh = KNeighborsClassifier(n_neighbors= 1, weights = "distance",metric="minkowski",p=1)
    
    ## Fit the model
    neigh.fit(train_image_feats, train_labels)
    
    ## Predict
    predicted_test_labels = neigh.predict(test_image_feats)


    return predicted_test_labels





def svm_classify(train_image_feats, train_labels, test_image_feats):

  #clf = SVC(random_state= 5, tol=1e-5, kernel ="linear",degree=3,break_ties=True)
  ##Linear svm (One vs rest)
  clf = LinearSVC(random_state= 0, tol=1e-3,C=5)

  ## Fit the model
  clf.fit(train_image_feats, train_labels)

  ## Predict
  predicted_test_labels = clf.predict(test_image_feats)

  return predicted_test_labels       
