import math
import glob
from textblob import TextBlob as tb
import nltk

sample = ''
path = '/Users/manjunathsindagi/Documents/redhat/python-projects/data/solr/*.txt'
#path = '/Users/manjunathsindagi/Documents/redhat/python-projects/data/test.txt'
files=glob.glob(path)
for file in files:
    f=open(file, 'r')
    str = f.read()
    sample = sample +" "+ str
   # print ('%s' % f.readlines())
    f.close()
 
#with open('/Users/manjunathsindagi/Documents/redhat/python-projects/data/solr/solr3.txt', 'r') as f:
 #   sample = f.read()


sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

def extract_entity_names(t):
    entity_names = []
   # print(t)	
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

entity_names = []
for tree in chunked_sentences:
    # Print results per sentence
    #print (extract_entity_names(tree))

    entity_names.extend(extract_entity_names(tree))

# Print all entity names
#print (entity_names)

freq_dist=nltk.FreqDist(entity_names)
#print(freq_nouns)
print(freq_dist.most_common(20))

# Print unique entity names
print (set(entity_names))
