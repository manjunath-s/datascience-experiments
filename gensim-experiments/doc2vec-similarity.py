from gensim.models import doc2vec
from gensim.models import Doc2Vec
from numpy import dot
from numpy.linalg import norm

# Reference : https://gist.github.com/balajikvijayan/9f7ab00f9bfd0bf56b14

def cosine( a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

sentence = doc2vec.LabeledSentence(
    words=[u'so`bme', u'words', u'here'], tags=["SENT_0"])
sentence1 = doc2vec.LabeledSentence(
    words=[u'here', u'we', u'go'], tags=["SENT_1"])

sentences = [sentence, sentence1]


model = Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(sentences)

for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save("../data/my_model.doc2vec")
model_loaded = Doc2Vec.load('../data/my_model.doc2vec')

print model.docvecs.most_similar(["SENT_0"])
print model_loaded.docvecs.most_similar(["SENT_1"])
print("cosine sim ", cosine(model.docvecs[0], model.docvecs[1]))