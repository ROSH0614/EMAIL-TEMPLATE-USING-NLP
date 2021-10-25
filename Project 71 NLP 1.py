#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk 
import numpy as np
import re
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('C://Users//Roshan//Downloads//Email_templates.csv')
df.head(20)


# In[3]:


import re

CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext


# In[4]:


def tags(s):
    s= str(s)
    s=s.replace("<p>","")
    s=s.replace("</p>","")
    s=s.replace("<br/>","")
    s=s.replace("</br>","")
    pattern = r"<p.{0,}>"
    s=re.sub(pattern,"",s)
    return s


# In[5]:


import re

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"",text)

def remove_html(text):
    html= re.compile(r"<.*?>")
    return html.sub(r"",text)


# In[6]:


df


# In[7]:


df["new_template"]= df.Template.apply(tags)


# In[8]:


new_template=df["new_template"]
new_template=new_template.to_frame()


# In[9]:


df.head(20)


# In[10]:


new_template.head(76)


# In[11]:


dff=new_template.head(76)


# In[12]:


dff


# In[13]:


dff.count()


# In[14]:


dff.ffill(axis = 0,inplace=True) # fills the null value with the previous value.
dff


# In[15]:


dff.count()


# In[16]:


dff.describe()


# In[17]:


df1=dff.head(10)


# In[18]:


def step1(x):
    for i in x:
        a=str(i).lower()
        p=re.sub(r'[^a-z0-9]',' ',a)
        print(p)


# In[19]:


step1(df1['new_template'])


# # cleaning the data

# In[20]:


import re #regular expression
import string

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)


# In[21]:


df['new_template'] = df.new_template.apply(clean)
df.new_template


# In[22]:


dff.head()


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(df['new_template'])
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']

wf_df


# In[24]:


import nltk
nltk.download('punkt')


# In[25]:


text = ' '.join(df['new_template'])
no_punc_text = text.translate(str.maketrans('', '', string.punctuation))
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
from nltk.probability import FreqDist
fdist = FreqDist(text_tokens)
print(fdist)


# In[26]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))

fdist.plot(100,cumulative=False)
plt.show()


# In[27]:


wf_df[0:10].plot.bar(x='word', figsize=(12,8), title='Stopwords in mails')


# In[28]:


import nltk
nltk.download('stopwords')


# # removal of stop words

# In[29]:


stop = stopwords.words('english')
new_words=('dear','name', 'work', 'email','please','would','time','company','sincerely','hi')
for i in new_words:
    stop.append(i)
print(stop)


# In[30]:


dff['CONTEXT_OF_SAMPLE_MAIL_without_stopwords'] = df['new_template'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[31]:


dff


# In[32]:


book = [x.strip() for x in dff.new_template] # remove both the leading and the trailing characters
book = [x for x in book if x]


# In[33]:


def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words)


# In[34]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[35]:


import nltk
nltk.download('wordnet')


# In[36]:


text_normalization('hi mike  my name is nick and i m a co founder at smart host  we help property managers optimize their pricing on marketplaces like homeaway  vrbo  and flipkey  i wanted to learn how you currently handle price optimization and show you what we re working on  are you available for a quick call tomorrow afternoon name   i d like to discuss your lead gen efforts  we re helping other  industry  companies collect their prospects straight from professional social networksand Import them directly into their crm  adding phone numbers And email addresses   quick question  can you put me In touch With whoever Is responsible For new prospecting And revenue  generating tools at  company   hi  name    myname  From  mycompany  here  companies make more sales With consistent marketing   mycompany  can put proven sales tools into the hands of everyone who sells your product  If that sound useful  i can explain how it works  hi  name   i m trying to figure out who Is In charge of  leading general statement  there at  company   would you mind pointing me towards the right person please  And the best way i might get In touch With them  hello  name   what would it mean to your top line revenue  you saw a   increase In contact rates    improvement In closes And   increase In quota hitting sales reps  let s find a few minutes to talk about how insidesales com Is providing these results to our clients  i m available tomorrow  insert 2 times you re available   can we sync up  hi  name   i hope this note finds you well  i ve been working For a company called  my company  that specializes In x  y And z  In thinking about your role at  company   i thought there might be a good fit For your group  our  product name  has garnered a lot of attention In the marketplace And i think it s something that your organization might see immediate value In  can you Help me get In contact With the right decision maker')


# In[37]:


df['lemmatized_text']=dff['CONTEXT_OF_SAMPLE_MAIL_without_stopwords'].apply(text_normalization) # applying the fuction to the dataset to get clean text
df.tail(15)


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X1 = cv.fit_transform(df['lemmatized_text'])
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df1 = pd.DataFrame(words_freq)
wf_df1.columns = ['word', 'count']

wf_df1


# In[39]:


wf_df1[0:10].plot.bar(x='word', figsize=(12,8), title='lemmatized text')


# # Frequency of words

# In[40]:


freq_Sw = pd.Series(' '.join(df['lemmatized_text']).split()).value_counts()[:50] # for top 20
freq_Sw


# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[42]:


text = df['lemmatized_text'].values 

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2').generate(str(text))
plot_cloud(wordcloud)


# In[43]:


data = df[["lemmatized_text","Template","Category"]]
data


# # named enitity recognition

# In[44]:


import spacy
nlp = spacy.load("en_core_web_sm")
text_nlp = nlp(no_punc_text)


# In[45]:


ner_tagged = [(word.text, word.ent_type_) for word in text_nlp]
from spacy import displacy

# visualize named entities
displacy.render(text_nlp, style='ent', jupyter=True)


# In[46]:


named_entities = []
temp_entity_name = ''
temp_named_entity = None
for term, tag in ner_tagged:
    if tag:
        temp_entity_name = ' '.join([temp_entity_name, term]).strip()
        temp_named_entity = (temp_entity_name, tag)
    else:
        if temp_named_entity:
            named_entities.append(temp_named_entity)
            temp_entity_name = ''
            temp_named_entity = None


# In[47]:


print(named_entities)


# In[48]:


from collections import Counter
c = Counter([item[1] for item in named_entities])
c.most_common()


# In[49]:


plt.rcParams["figure.figsize"] = [14, 6]
plt.rcParams["figure.autolayout"] = True
plt.bar(c.keys(), c.values())

plt.show()


# # part of speech tagging

# In[50]:


for token in text_nlp:
    print(token, token.pos_)


# In[51]:


#Filtering for nouns and verbs only
nouns_verbs = [token.text for token in text_nlp if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs[0:50])#Filtering for nouns and verbs only


# In[52]:


import nltk
nltk.download('averaged_perceptron_tagger')
pos_tags = nltk.pos_tag(text_tokens)
pd.DataFrame(pos_tags).T


# In[53]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count'] = df['lemmatized_text'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['lemmatized_text'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['lemmatized_text'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['lemmatized_text'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['lemmatized_text'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[54]:


df


# # Bigram analysis

# In[55]:


def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[56]:


top2_words = get_top_n2_words(df['lemmatized_text'], n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df.head()


# # bigram visualization

# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])


# # Trigram analysis

# In[58]:


def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[59]:


top3_words = get_top_n3_words(df['lemmatized_text'], n=200)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]


# In[60]:


top3_df


# # trigram visualization

# In[61]:


import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])


# # LABELING THE OUTPUT

# # MODEL BUILDING

# # Cosine Similarity 

# In[62]:


df.new_template[0]


# In[63]:


from sklearn.metrics.pairwise import cosine_similarity


# In[64]:


countvectorizer = CountVectorizer(stop_words='english')
sparse_matrix = countvectorizer.fit_transform(df['new_template'])


# In[65]:


sent = countvectorizer.transform(['hi helped a startup'])


# In[66]:


df['similarity']= cosine_similarity(sparse_matrix,sent)


# In[67]:


df[['similarity','new_template']].sort_values(by=['similarity'],ascending=False).head(10)


# In[68]:


data


# In[ ]:





# # TFIDF VECTORIZER

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[70]:


vectorizer = TfidfVectorizer(analyzer='word',norm=None, use_idf=True,smooth_idf=True)
tfIdfMat  = vectorizer.fit_transform(df)


# In[71]:


feature_names = sorted(vectorizer.get_feature_names())


# In[72]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(df['lemmatized_text'])


# In[73]:


X = tfidf.transform(df['lemmatized_text'])
df['lemmatized_text'][1]


# In[74]:


print([X[1, tfidf.vocabulary_['marketplace']]])


# In[75]:


print([X[1, tfidf.vocabulary_['optimization']]])


# In[76]:


print([X[1, tfidf.vocabulary_['homeaway']]]) # rarer word than the two 


# In[117]:


from sklearn.model_selection import train_test_split
X = df.new_template
y = df.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),
                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))


# In[118]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


# In[119]:


cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,30001,10000)
def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Test result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result
tfidf = TfidfVectorizer()
print("Result for trigram with stop words (Tfidf)\n")
v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
x = v.fit_transform(df['new_template'].values.astype('U'))
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))


# In[120]:


df['Template'] = df['new_template'].factorize()[0]
from io import StringIO
category_id_df = df[['new_template', 'Template']].drop_duplicates().sort_values('Template')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Template', 'new_template']].values)


# In[ ]:





# # Visualization 

# In[122]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,10))
df.groupby('Category').lemmatized_text.count().plot.bar(ylim=0)
plt.show()


# # TF-IDF

# In[123]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.lemmatized_text).toarray()
labels = df.new_template
features.shape


# # PERFORMING AND VISUALIZE NGRAM ANALYSIS ON EACH TEMPLATE

# In[124]:


from sklearn.feature_selection import chi2
import numpy as np

N = 2
for Template, new_template in sorted(new_template.items()):
  features_chi2 = chi2(features, labels == new_template)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  print("# '{}':".format(new_template))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))


# # SPLITING INTO TRAIN AND TEST

# # FIT'S WITH CountVectorizer and TfidfTransformer

# In[125]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['Template'], df['new_template'],test_size=0.26, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform((X_train).values.astype('U'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_counts = count_vect.fit_transform((X_test).values.astype('U'))
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)


# In[126]:


X_train.shape


# In[127]:


X_test.shape


# In[128]:


X_train_tfidf.shape


# In[129]:


X_test_tfidf.shape


# # USING MULTIPLE MODEL 

# In[130]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[131]:


get_ipython().system('pip install xgboost')


# In[132]:


X1=df['new_template']
Y1=df['Category']
x = df['new_template']
y = df['Category']
vect = CountVectorizer()


# In[133]:


x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,test_size=0.05,random_state=42)
x_train_dtm = vect.fit_transform((x_train1).values.astype('U'))
x_test_dtm = vect.transform((x_test1).values.astype('U'))


# # Naive Bayes

# In[134]:


NB = MultinomialNB()
NB.fit(x_train_dtm,y_train1)
y_predict = NB.predict(x_test_dtm)
NB_acc = metrics.accuracy_score(y_test1,y_predict)
NB_acc


# # SGD Classifier

# In[135]:


from sklearn.linear_model import LinearRegression 
lm = SGDClassifier()
lm.fit(x_train_dtm,y_train1)
lm_predict = lm.predict(x_test_dtm)
SGD_acc= metrics.accuracy_score(y_test1,lm_predict)
SGD_acc


# # Random Classifier

# In[136]:


rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(x_train_dtm,y_train1)
rf_predict = rf.predict(x_test_dtm)
Random_acc = metrics.accuracy_score(y_test1,rf_predict)
Random_acc


# # XGBoost

# In[137]:


xg = XGBClassifier()
xg.fit(x_train_dtm,y_train1)
xg_predict = xg.predict(x_test_dtm)
xg_acc = metrics.accuracy_score(y_test1,xg_predict)
xg_acc


# # Passive Agressive Classifier

# In[138]:


pg = PassiveAggressiveClassifier()
pg.fit(x_train_dtm,y_train1)
pg_predict = pg.predict(x_test_dtm)
pg_acc = metrics.accuracy_score(y_test1,pg_predict)
pg_acc


# # Linear SVC

# In[139]:


lv = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', tol=0.0001,
     verbose=0)
lv.fit(x_train_dtm,y_train1)
lv_predict = lv.predict(x_test_dtm)
lv_acc= metrics.accuracy_score(y_test1,lv_predict)
lv_acc


# In[140]:


data = {'Model':['MultinomialNB','SGDClassifier','RandomForestClassifier','XGBClassifier', 'PassiveAggressiveClassifier','LinearSVC'],'Accuracy':[NB_acc, SGD_acc,Random_acc,xg_acc,pg_acc,lv_acc]}


# # Accuracy score of each model

# In[141]:


data_model = pd.DataFrame(data)
data_model.sort_values(by=['Accuracy'], ascending=False)


# In[142]:


import seaborn as sns
plt.figure(figsize=(15,8))
sns.boxplot(x='Model', y='Accuracy', data=data_model)
sns.stripplot(x='Model', y='Accuracy', data=data_model, 
              size=8, jitter=True, edgecolor="gray", linewidth=2,)
plt.show()


# In[143]:


data_model.plot(x="Model", y="Accuracy", kind="bar")


# # FINALIZE THE MODEL USING PassiveAggressiveClassifier

# In[146]:


X_COUNT = count_vect.fit_transform(X1)
X_TFIDF = tfidf_transformer.fit_transform(X_COUNT)


# In[147]:


modelf=PassiveAggressiveClassifier().fit(X_TFIDF, Y1)


# # SAMPLE PREDICTION OF FINAL MODEL

# In[148]:


print(modelf.predict(count_vect.transform(["appoinment schedule"])))


# In[149]:


print(modelf.predict(count_vect.transform(["hi my name is nick i want your shoes"])))


# In[150]:


print(modelf.predict(count_vect.transform(["job search"])))


# # Saving the model in the pickle file

# In[151]:


import pickle


# In[152]:


modelf=PassiveAggressiveClassifier().fit(X_TFIDF, Y1)

    # Save the vectorizer
vec_file = 'vectorizer.pickle'
pickle.dump(count_vect, open(vec_file, 'wb'))

    # Save the model
mod_file = 'classification.model'
pickle.dump(modelf, open(mod_file, 'wb'))


# # our model works good with Multinomial Classifier

# In[ ]:




