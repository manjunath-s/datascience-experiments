from gensim.models import doc2vec
from gensim.models import KeyedVectors
import gensim
from collections import namedtuple
import math
from numpy import dot
from numpy.linalg import norm


#Gensim Test.
# Load data
# #
# http://stackoverflow.com/questions/31321209/doc2vec-how-to-get-document-vectors
# http://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

class textcompare():
    def __init__(self):
        pass

    def rungensim(self,docs):

        # Transform data (you can add more data preprocessing steps)

        docs = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i, text in enumerate(doc2):
            words = text.lower().split()
            #print(words)
            tags = [i]
            docs.append(analyzedDocument(words, tags))

        model = doc2vec.Doc2Vec(docs, size = 3, window = 30, min_count = 2, workers = 4)
        # Get the vectors
        #print(model.docvecs[0])
        #print(model.docvecs[1])
        print("infer ",model.infer_vector("Chattisgarh, CRPF"))
        #compute cosine
        print("cosine sim ", self.cosine(model.docvecs[0], model.docvecs[1]))


    def cosine(self,a,b):
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        return cos_sim

    def jaccard(self,doc_list1, doc_list2):
        set_1 = set(doc_list1)
        set_2 = set(doc_list2)
        n = len(set_1.intersection(set_2))
        sim = n / float(len(set_1.union(set_2)))
        print("jaccard",sim)


if __name__ == "__main__":
    doc1 = ["This is a sentence", "This is another sentence"]
    doc2 = ["11 CRPF personnel killed in naxal attack at Chattisgarh",
            "11 CRPF personnel killed in encounter with Naxal in Chhattisgarh Sukma"]

    doc1_list = doc1[0].lower().split(" ")
    doc2_list = doc1[1].lower().split(" ")
    g = textcompare()
    g.rungensim(doc1)
    g.jaccard(doc1_list,doc2_list)
