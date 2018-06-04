
import gensim
from gensim import corpora, models
from gensim.summarization import summarize, keywords

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Plotting tools
#import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

from os import path, listdir
import sys
import json
import random
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

en_nlp = spacy.load('en')
es_nlp = spacy.load('es')

STOP_WORDS.add("'s")

my_project_dir = path.dirname(__file__)
en_file = path.join(my_project_dir + "data/" + "en_example.json")
es_file = path.join(my_project_dir + "data/" + "es_example.json")
#fh_en = open("en_topics_results.txt", 'w')
#fh_es = open("es_topics_results.txt", 'w')

def transform(text, lang='en'):
	tokens = []
	if lang == 'en':
		doc = en_nlp(text)
	elif lang == 'es':
		doc = es_nlp(text)
	for token in doc:
		if not token.is_punct and not token.is_stop and len(token.text)>2:
			tokens.append(token.text)
	return tokens

def model(file):
	with open(file, 'r') as fp:
		json_data = json.load(fp)
		texts = []
		for line in json_data:
		# apply nlp pipeline to texts per paragraph
			text = transform(line['p_text'])
			texts.append(text)

			#turn our tokenized documents into a id <-> term dictionary
		dictionary_doc = corpora.Dictionary(texts)
		# gensim corpus
		corpus = [dictionary_doc.doc2bow(text) for text in texts]
		# applying a model
		ldamodel_doc = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary_doc)
		print("Topics: " + str(ldamodel_doc.print_topics(num_topics=5, num_words=2)) + "\n")
		#vis = pyLDAvis.gensim.prepare(ldamodel_doc, corpus, dictionary_doc)
		#pyLDAvis.display(vis)

model(en_file)
model(es_file)

'''
	with open(my_project_dir + "data/" + es_file, 'r') as fp:
	json_data = json.load(fp)
	texts = []
	for line in json_data:
		# apply nlp pipeline to texts per paragraph
		text = transform(line['p_text'], 'es')
		texts.append(text)

	#turn our tokenized documents into a id <-> term dictionary
	dictionary_doc = corpora.Dictionary(texts)
	# gensim corpus
	corpus = [dictionary_doc.doc2bow(text) for text in texts]
	# applying a model
	ldamodel_doc = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary_doc)
	print("Topics: " + str(ldamodel_doc.print_topics(num_topics=5, num_words=2)) + "\n")
'''
