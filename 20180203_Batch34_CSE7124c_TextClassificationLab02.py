import os 
import re  
import pandas as pd
from sklearn.metrics import confusion_matrix 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics 
import nltk


os.chdir("F:\insofe data\Rstudio\lab30_3_2_18__page rank,clustering,navie bayes")

##### Reading data using Pandas and preparaing data 
short_pos=pd.read_table("short_reviews\\positive.txt",sep="\n", header=None,encoding='latin-1')
short_neg=pd.read_table("short_reviews\\negative.txt",sep="\n", header=None,encoding='latin-1')

short_pos.columns =["Reviews"]
short_neg.columns =["Reviews"]

short_pos["Sentiment"]="1"
short_neg["Sentiment"]="0"

x_short_pos=short_pos[:1000]
x_short_neg=short_neg[:1000]
##### Combining the Positive and negative Reviews

data=pd.concat([x_short_pos, x_short_neg])
data.index=range(len(data.Sentiment))

#### deleting unnecesarry variables

del(short_neg, short_pos, x_short_neg, x_short_pos)

##### Part 1 Preprocessing and Buliding SVM

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
  
    
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
    
####raw_review= data["Reviews"].iloc[0]
    
num_reviews = data["Reviews"].size

# Initialize an empty list to hold the clean reviews
clean_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_reviews.append( review_to_words( data["Reviews"][i] ) )


vectorizer = CountVectorizer(analyzer = "word",\
                             tokenizer = None,\
                             preprocessor = None,\
                             stop_words = None) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
data_features = vectorizer.fit_transform(clean_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
data_features = data_features.toarray()

data_features=pd.DataFrame(data_features)

data_features["Sentiment"]=data["Sentiment"]

data_features = data_features.sample(frac =1)

### Splitting data into train & test

train,test = train_test_split(data_features, test_size = 0.2) 

cols = [col for col in data_features.columns if col not in ["Sentiment"]]

train.x= train[cols]
train.y=train["Sentiment"]

test.x=test[cols]
test.y=test["Sentiment"]

## bulding SVM 

model_linear = svm.SVC(kernel='linear') 
model_linear.fit(train.x,train.y.astype(int)) 
preds = model_linear.predict(test.x)
confusion_matrix(test.y.astype(int),preds)

accuracy = metrics.accuracy_score(test.y.astype(int), preds)
print(accuracy)


model_rbf = svm.SVC(kernel='rbf',C=3,gamma=0.02) 
model_rbf.fit(train.x,train.y.astype(int)) 
preds1 = model_rbf.predict(test.x)
confusion_matrix(test.y.astype(int),preds1)

accuracy = metrics.accuracy_score(test.y.astype(int), preds1)
print(accuracy)

############ Taking only adjectives and then bulding model 

allowed_word_types = ["J"]            

def review_to_Pos( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    all_words = []
    
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. tag the reveiws
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            # extracting the adjectives
            all_words.append(w[0])
     
    #4. remove stopwords
    stops = set(stopwords.words("english"))     
    
    all_words = [w for w in all_words if not w in stops] 
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(all_words))
    
    
#raw_review= data["Reviews"].iloc[0]

num_reviews = data["Reviews"].size

# Initialize an empty list to hold the clean reviews
clean_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_reviews.append( review_to_Pos( data["Reviews"][i] ) )


vectorizer = CountVectorizer(analyzer = "word",\
                             tokenizer = None,\
                             preprocessor = None,\
                             stop_words = None) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
data_features = vectorizer.fit_transform(clean_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
data_features = data_features.toarray()

data_features=pd.DataFrame(data_features)

data_features["Sentiment"]=data["Sentiment"]

data_features = data_features.sample(frac =1)

### Splitting data into train & test

train,test = train_test_split(data_features, test_size = 0.2) 

cols = [col for col in data_features.columns if col not in ["Sentiment"]]

train.x= train[cols]
train.y=train["Sentiment"]

test.x=test[cols]
test.y=test["Sentiment"]

## bulding SVM 

model_linear = svm.SVC(kernel='linear') 
model_linear.fit(train.x,train.y.astype(int)) 
preds = model_linear.predict(test.x)
confusion_matrix(test.y.astype(int),preds)

accuracy = metrics.accuracy_score(test.y.astype(int), preds)
print(accuracy)


model_rbf = svm.SVC(kernel='rbf',C=3,gamma=0.02) 
model_rbf.fit(train.x,train.y.astype(int)) 
preds1 = model_rbf.predict(test.x)
confusion_matrix(test.y.astype(int),preds1)

accuracy = metrics.accuracy_score(test.y.astype(int), preds1)
print(accuracy)



#####

