
# coding: utf-8

# # Clustering Job Titles

#     This document explains how to cluster list of job titles.
#     Steps I have followed
#         1.Taking random sample from the list of jobs and  tokenizing each title
#         2.Vectorizing the titles using tf-idf vectorization technique
#         3.calculate distance b/n each titles and create distance matrix
#         4.Find optimal k using silhouette scores for each sample
#         5.cluster using k-means Algorithm
#         6.Reduce dimentionality using
#         7.Plot k-mean cluster output
#         8.Apply hierarchical clustering on the titles using Ward clustering
#         9.plot the dendogram
#

# ###### Lets import libraries needed up front

# In[388]:
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3


# #### Loading the csv dataset and taking random 1000 samples from the dataset and data preprocessing

# In[450]:
nltk.download('stopwords')
nltk.download('punkt')

dataset = pd.read_csv('person-titles.csv')
titles = dataset['title'].dropna( how='any')
titles = titles.sample(n=1000)#taking 100 random sapmles from the data

print(titles.head(100))
#print titles[:10] #first 10 titles
stopwords = nltk.corpus.stopwords.words('english')
print(len(titles.unique()))


# This section is focused on defining some functions to manipulate the Job titles. First, I load NLTK's list of English stop words. Stop words are words like "a", "the", or "in" which don't convey significant meaning. I'm sure there are much better explanations of this out there.

# In[451]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[452]:

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[453]:


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in titles:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# ## Tf-idf and document vectorizer¶

# Here, I define term frequency-inverse document frequency (tf-idf) vectorizer parameters and then convert the titles list into a tf-idf matrix.To get a Tf-idf matrix, first count word occurrences by document(i,e word occurrences in each job title). This is transformed into a document-term matrix (dtm). This is also just called a term frequency matrix. Words that occur frequently within a document but not frequently within the corpus receive a higher weighting as these words are assumed to contain more meaning in relation to the document.
#     max_df: this is the maximum frequency within the documents a given feature can have to be used in the tfi-idf matrix. If the term is in greater than 80% of the documents it probably cares little meanining (in the context of job title)
#     min_idf: this could be an integer (e.g. 5) and the term would have to be in at least 5 of the documents to be considered. Here I pass 1; the term must be in at least 1 of the document. I found that if I allowed a lower min_df I ended up basing clustering on names--for example " Sr." or "Director" are names found in several of the job titles and the job titles use these names frequently, but the names carry no real meaning.
#

# In[527]:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=800, max_features=1000,
                                 min_df=1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


tfidf_matrix = tfidf_vectorizer.fit_transform(titles) #fit the vectorizer to job titltes

print(tfidf_matrix.shape)



terms = tfidf_vectorizer.get_feature_names()



# In[455]:

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)# calculating distances b/the titles


# ### Finding the best k value in this case of sample data taken it looks like 18 has shown better clustring performance rather than other range of numbers I have given

# In[468]:



from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.


range_n_clusters = [9,10]
for n_clusters in range_n_clusters:

    for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns


        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(tfidf_matrix)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tfidf_matrix, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values =                 sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i




#
# Cluster the sample data into 18 clusters

# In[456]:

from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


# Here is some fancy indexing and sorting on each cluster to identify which are the top n (I chose n=7) words that are nearest to the cluster centroid. This gives a good sense of the main topic of the cluster.

# In[474]:


jobs = { 'title': titles,  'cluster': clusters }

frame = pd.DataFrame(jobs,  columns = [ 'title','cluster'])
frame['cluster'].value_counts()#number of jobs clustered (0-9)


# In[469]:

Words = []
word = []
for i in range(num_clusters):
    for ind in order_centroids[i, :7]: #replace 6 with n words per cluster
        word.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
cluster_names = []
j=1
s =len(word)/7
print(s)
for i in range(s):
    k = i*7
    l = j*7
    cluster_names.append(word[k:l])
    j=j+1
len(word)


# In[484]:


print("Top terms per cluster:")
#print()
#sort cluster centers by proximity to centroid

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
cluster_names = pd.DataFrame(cluster_names)
j=0
for i in range(num_clusters):
    for ind in order_centroids[i, :7]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print() #add whitespace
        print() #add whitespace
    print("Cluster %d titles:" % i, end='')
    for title in cluster_names.ix[i].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()
cluster_names = cluster_names.values.tolist()


# Generate random 20 colors for visualizing the 20 clusters and arranging the cluster words for visualization.

# In[485]:

import colorsys

def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out
cluster_colors = get_N_HexCol(num_clusters)
#cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

print(len(cluster_names))
print (len(cluster_colors))


# Here is some code to convert the dist matrix into a 3-dimensional array using multidimensional scaling.

# In[486]:

#print(word[0:10])
import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert three components as we're plotting points in a three-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys,zs = pos[:, 0], pos[:, 1],pos[:,2]
print(len(word))




# In[487]:

df = pd.DataFrame(dict(x=xs, y=ys,z =zs ,label=clusters, title=titles))
groups = df.groupby('label')


# As you we can observe the data in three dimentional view we have to many overlaps and clearly we do have to many noises to deal with.

# In[488]:

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(xs, ys, zs,c='red',
           cmap=plt.cm.coolwarm)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])


ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

plt.show()


# In[489]:

plt.scatter(xs,ys,c='black')
plt.show()


# For better visualization I have used D3.js its user interactive you can zoom in and out and hover mouse to see jobs titles.

# In[490]:

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 950);
      this.fig.toolbar.toolbar.attr("y", 800);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


# In[491]:

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot
fig, ax = plt.subplots(figsize=(16,12)) #set plot size
ax.margins(0.1) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                      label=list(set(cluster_names[name])),mec='none',
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=3, hoffset=1, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot


# In[492]:

from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_matrix = linkage(dist,'ward') #define the linkage_matrix using ward clustering pre-computed distances




# In[493]:

print(len(linkage_matrix))
print(len(dist))
len(titles)
dist.shape
linkage_matrix.shape


#
#
# As we have discussed during the onsite interview here is another way of clustering algorithm which is agglomerative clustering method, meaning that at each stage, the pair of clusters with minimum between-cluster distance are merged. I used the precomputed cosine distance matrix (dist) to calculate a linkage_matrix, which I then plot as a dendrogram.This type of clustering can give you better over vie Here we can observe that it has returned two primary clusters with the largest cluster being split into about 10 major subclusters.The major problem observed is that as you can see the Senior Accountant, senior Enginner..etc all the senior positions are clustered into one class.Which basically can be understood from the concept of text similarity.
#
# Note: Please zoom out to see the text.
#

# In[526]:

fig = plt.figure(figsize=(15,200))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linkage_matrix, ax=ax,labels=titles.values.tolist(),orientation="left")
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=8)
fig.savefig('t.png')

#The horizontal axis of the dendrogram represents the distance or dissimilarity between clusters. The vertical axis
#represents the objects and clusters. The dendrogram is fairly simple to interpret. Remember that our main interest
#is in similarity and clustering. Each joining (fusion) of t
#wo clusters is represented on the graph by the splitting of
#a horizontal line into two horizontal lines. The horizontal position of the split, shown by the short vertical bar,
#gives the distance (dissimilarity) between the two clusters.
