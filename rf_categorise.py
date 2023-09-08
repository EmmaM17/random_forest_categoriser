import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

# Text modification/cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import FreqDist

# ML/SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

url = 'ManuallyCategorisedList - Copy.xlsx'
df = pd.read_excel(url, usecols='B,D')

#### Clean Data ####
#Replacing any empty cells with 'no comment', then removing them all from our training data
df['Comment'].fillna('no comment', inplace=True)
df['Category'].fillna('no comment', inplace=True)
df = df[df['Category'] != 'no comment']

data = list(df['Comment'])
labels = list(df['Category'])
new_labels=[]
for label in labels:
    # Split the string at the hyphen
    label = str(label)
    split_string = label.split('-')
    # Save the part of the string before the hyphen
    part_before_hyphen = split_string[0]
    part_before_hyphen=part_before_hyphen.rstrip() #remove any blank spaces after word
    new_labels.append(part_before_hyphen)

pd.set_option('display.max_rows', None) #allows you to see the whole dataframe in the terminal

# Make data lowercase and labels capitalized for consistency
data = [entry.lower() for entry in data]
labels = [entry.capitalize() for entry in new_labels]

# Tokenize Data
data = [word_tokenize(entry) for entry in data]
   
#simplify sentences by removing unnecessary words like 'the'
word_lemmatized = WordNetLemmatizer()
filtered_data = []
for index, entry in enumerate(data):
    final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word)
            final_words.append(word_final)
    filtered_data.append(' '.join(final_words))

# Create cleaned DataFrame -shows simplified data and it's category
cleaned = pd.DataFrame({'Test Data': filtered_data, 'Actual Label': labels})

#### Find the most common words or phrases in each category ####
#for every category find the 15 most common words used
def find_common_phrases(cleaned, category):
    dataset = cleaned[cleaned['Actual Label'] == category]
    
    word_list=[]
    for index, row in dataset.iterrows():
        words= row['Test Data']
        words = nltk.word_tokenize(words)
        word_list.extend(words)

    freq_dist = FreqDist(word_list)
    common_phrases = freq_dist.most_common(15) #increasing the number could improve accuracy???
    top_words=[]
    for phrase, frequency in common_phrases: #use frequency to tell you how many times the word appears
        top_words.append(phrase)
    
    return top_words #returns 15 most common words for a given category

categories = ['Document control', 'Null', 'Quality', 'Technical', 'Risk', 'Uncategorised']
all_top_words=[]
top_word_df=pd.DataFrame()

for category in categories:
    category_top_words=find_common_phrases(cleaned, category) #stores find_common_phrase output
    all_top_words.extend(category_top_words) #adds common words to list of all common words
    category_common_words=pd.DataFrame({'Category':category, 'most common words':category_top_words})
    top_word_df=pd.concat([top_word_df,category_common_words], ignore_index=True) #dataframe that stores common words and where they came from
    

all_top_words = list(set(all_top_words))

#now remove any words that appear in more than one category
# Drop rows with duplicated values in the "most common words" column
top_word_df = top_word_df[~top_word_df['most common words'].duplicated()]
# Verify if duplicates have been removed
duplicates_exist = top_word_df['most common words'].duplicated().any()

#store each categories key words
document_control_words = top_word_df[top_word_df['Category'] == 'Document control']['most common words'].tolist()
null_words = top_word_df[top_word_df['Category'] == 'Null']['most common words'].tolist()
quality_words = top_word_df[top_word_df['Category'] == 'Quality']['most common words'].tolist()
technical_words = top_word_df[top_word_df['Category'] == 'Techincal']['most common words'].tolist()
risk_words = top_word_df[top_word_df['Category'] == 'Risk']['most common words'].tolist()
uncategorised_words = top_word_df[top_word_df['Category'] == 'Uncategorised']['most common words'].tolist()


#### Categorising ####

#Now we should be able to predict the category by first checking if the test data contains keywords
#If it does then it has a high probabilty of being in a certain category
#If it doesn't contain a key word then we should use NLP to attempt to categorise it.

#### Training ####
#x = data y = labels
X_train, X_test, y_train, y_test = train_test_split(filtered_data, new_labels, test_size=0.8)

#Now check whether the data contains key phrases

def word_check(words,X_test,category,y_test):
    for word in words:
        if word in X_test:
            print(word)
            print ('found in category: ',category)
            
            newPrediction=pd.DataFrame({'Actual label':y_test,'Test Data': X_test, 'Prediction': word})
            X_test.remove(word)
            y_test.remove

            #!!!Need to remove test data and it's label from the testing pool ^^
    return X_test, y_test, newPrediction
            

#Should check to see if we can match any words within test data to a category 
#For now it's working on the assumption that if it finds the word then it is definitely part of that category     
prediction=pd.DataFrame()
for category in categories:  
    X_test, y_test, newPrediction=word_check(document_control_words,X_test,category,y_test) #replace category with 'Risk' to test
    prediction=pd.concat([prediction,newPrediction], ignore_index=True) #we will add NLP predictions later to give us our over all accuracy

#The following code should train a random forest model with the remaining data
""" 
# Vectorize
Tfidf_vect = TfidfVectorizer(max_features=6000)
Tfidf_vect.fit(filtered_data)  # <- remove the data we already categorised(^) from filtered_data 

# Transform text data
X_train_tfidf = Tfidf_vect.transform(X_train)
X_test_tfidf = Tfidf_vect.transform(X_test)

# Random forest
rf = RandomForestClassifier(n_estimators=100) # <- Increase for higher accuracy???
rf.fit(X_train_tfidf, y_train)
 """

#### Prediction ####

# Generate random forest predictions using the test data
#rf_predictions = rf.predict(X_test_tfidf)
#rf_predictions_df = pd.DataFrame({'Actual Category': y_test, 'Data': X_test, 'Prediction': predictions})

## Now we need to combine the manual categorised predictions and random forest ones
#prediction = pd.concat([prediction,rf_predictions_df],ignore_index = True)
#accuracy = accuracy_score(prediction[Prediction], y_test) * 100

####  Display  Output ####
""" #confusion matrix
def confusion_matrix_plot(y_test,predictions):
    cm = confusion_matrix(y_test,predictions,labels=SVM.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=SVM.classes_,)
    disp.plot(xticks_rotation=30, cmap='Oranges')
    plt.show()

confusion_matrix_plot(y_test,predictions) """