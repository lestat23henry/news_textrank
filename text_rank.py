#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
from collections import defaultdict
import numpy as np
from jieba._compat import *
import math

from pathlib import Path
import json
import codecs

import gensim.models.word2vec as w2v
from collections import Counter
from multiprocessing import cpu_count

import sqlite3
import split_word

#import vector_clustering

CORPORA_PATH = '/home/lc/ht_work/data/all_news_5_bak'
TMPWORDFILE = '/tmp/tmpwords'
MODELPATH_3 = "/home/lc/ht_work/news_textrank/news3_model"

#------- method 1 ---------------
WEIGHT_COV = 0.4
WEIGHT_LOC = 0.6
WEIGHT_POS = 0.0
WEIGHT_FREQ = 0.0
LAMBDA_TITLE = 60
LAMBDA_FIRST = 10
LAMBDA_LAST = 10

#------- method 2 ---------------
WEIGHT_W2V = 0.5
WEIGHT_ORG = 0.5
MODEL_PATH = "/home/lc/ht_work/data/200_0_0_1/news.model"

#------- method 0 ---------------
WEIGHT_LOC_RAND = 1.0
WEIGHT_POS_RAND = 0.0

class UndirectWeightedGraph:
	d = 0.85
	model = None

	def __init__(self,method=0):
		self.graph = defaultdict(list)

		if method == 0:
			self.pos_weight = WEIGHT_POS_RAND
			self.loc_weight = WEIGHT_LOC_RAND
		elif method == 1:
			self.pos_weight = WEIGHT_POS
			self.loc_weight = WEIGHT_LOC
			self.freq_weight = WEIGHT_FREQ
			self.cov_weight = WEIGHT_COV

	def addEdge(self, start, end, weight):
		# use a tuple (start, end, weight) instead of a Edge object
		self.graph[start].append((start, end, weight))
		self.graph[end].append((end, start, weight))

	@classmethod
	def set_model(cls,modelpath):
		try:
			print 'load word2vec model at %s start!\n' % modelpath
			cls.model = w2v.Word2Vec.load(modelpath)
			print 'load word2vec model at %s end!\n' % modelpath
		except Exception,e:
			print e.message

		return True

	def _rand_method_0(self,preference):
		total_pref = {}
		#calc final preference
		if preference:
			for w in list(set(self.graph.keys())):
				total_pref[w]= self.pos_weight * preference['pos'][w] + self.loc_weight * preference['loc'][w]

		ws = defaultdict(float)
		outSum = defaultdict(float)

		wsdef = 1.0 / (len(self.graph) or 1.0)
		for n, out in self.graph.items():
			ws[n] = wsdef
			outSum[n] = sum((e[2] for e in out), 0.0)

		pref={}
		if total_pref:
			s=float(sum(total_pref.values()))
			for k,v in total_pref.items():#偏好概率归一化
				pref[k]=float(v)/float(s)


		sorted_keys = sorted(self.graph.keys())
		for x in xrange(30):  # 10 iters
			for n in sorted_keys:
				s = 0
				for e in self.graph[n]:
					s += e[2] / outSum[e[1]] * ws[e[1]]
				if not pref:
					ws[n] = (1 - self.d) + self.d * s
				else:
					ws[n] = (1 - self.d)*pref[n] + self.d * s

		return ws

	def _rank_method_1(self,preference):
		ws = defaultdict(float)

		w_cov = self.cov_weight
		w_loc = self.loc_weight
		w_pos = self.pos_weight
		w_freq = self.freq_weight

		w_mat = {}

		if preference:
			pref_loc = preference['loc']
			pref_pos = preference['pos']
			pref_freq = preference['freq']

			wsdef = 1.0 / (len(self.graph) or 1.0)
			for i,out in self.graph.items():
				ws[i] = wsdef

			sumi = {}
			sumc = {}
			sump = {}
			for i,out in self.graph.items():
				sumi[i] = 0.0
				sumc[i] = 0.0
				sump[i] = 0.0
				for e in out:
					sumi[i] += pref_loc[e[1]]
					sumc[i] += pref_freq[e[1]]
					sump[i] += pref_pos[e[1]]

			for i,out in self.graph.items():
				w_mat[i] = {}
				for e in out:
					w_mat[i][e[1]] = 0.0
					w_mat[i][e[1]] += w_cov * 1.0 / len(out)
					w_mat[i][e[1]] += w_loc * pref_loc[e[1]] / sumi[i]
					w_mat[i][e[1]] += w_freq * pref_freq[e[1]] / sumc[i]
					w_mat[i][e[1]] += w_pos * pref_pos[e[1]] / sump[i]

			sorted_i_keys = sorted(w_mat.keys())
			# w_mat 转移矩阵概率归一化
			for i in sorted_i_keys:
				s = 0.0
				sorted_j_keys = sorted(w_mat[i].keys())
				for j in sorted_j_keys:
					s += w_mat[i][j]
				for j in sorted_j_keys:
					w_mat[i][j] = w_mat[i][j] / s


			sorted_keys = sorted(self.graph.keys())
			for x in xrange(20):  # 10 iters
				for n in sorted_keys:
					s = 0.0
					for e in self.graph[n]:
						s += w_mat[e[1]][n] * ws[e[1]]

					ws[n] = (1 - self.d) + self.d * s

		return ws

	def _rank_method_2(self,preference):
		ws = defaultdict(float)
		outSum = defaultdict(float)
		outSum_sim = defaultdict(float)

		wsdef = 1.0 / (len(self.graph) or 1.0)
		for n, out in self.graph.items():
			ws[n] = wsdef
			outSum_sim[n] = 0.0
			outSum[n] = sum((e[2] for e in out), 0.0)
			for e in out:
				try:
					outSum_sim[n] += self.model.similarity(n,e[1])
				except Exception,e1:
					#print e1.message
					outSum_sim[n] += 0.5
					continue

		w_mat = {}
		for n, out in self.graph.items():
			w_mat[n] = {}
			for e in out:
				try:
					if n == u'ARM':
						print 'ARM'
					w_mat[n][e[1]] = 0.0
					w_mat[n][e[1]] += WEIGHT_W2V * self.model.similarity(n,e[1]) / outSum_sim[n]
					w_mat[n][e[1]] += WEIGHT_ORG * e[2] / outSum[n]
				except Exception,e1:
					#print e1.message
					w_mat[n][e[1]] += WEIGHT_W2V * 0.5 / outSum_sim[n]
					w_mat[n][e[1]] += WEIGHT_ORG * e[2] / outSum[n]
					continue

		sorted_i_keys = sorted(w_mat.keys())
		# w_mat 转移矩阵概率归一化
		for i in sorted_i_keys:
			s = 0.0
			sorted_j_keys = sorted(w_mat[i].keys())
			for j in sorted_j_keys:
				s += w_mat[i][j]
			for j in sorted_j_keys:
				w_mat[i][j] = w_mat[i][j] / s

		sorted_keys = sorted(self.graph.keys())
		for x in xrange(30):  # 10 iters
			for n in sorted_keys:
				s = 0.0
				for e in self.graph[n]:
					s += w_mat[n][e[1]] * ws[e[1]]

				ws[n] = (1 - self.d) + self.d * s

		return ws

	def rank(self,preference=None,method=0):
		# method : 0  -->  use
		result = None
		if method == 0 or method == -1:
			result = self._rand_method_0(preference)
		elif method == 1:
			result = self._rank_method_1(preference)
		elif method == 2:
			result = self._rank_method_2(preference)


		(min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

		for w in itervalues(result):
			if w < min_rank:
				min_rank = w
			if w > max_rank:
				max_rank = w

		for n, w in result.items():
			# to unify the weights, don't *100.
			result[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

		return result

class TextRank():

	pos_legal = ("n","ns","nt","nz","a","an","vn","i","I","j")
	stop_words = []

	def __init__(self):
		self.span = 5

		self.pos_pref = {}
		self.loc_pref = {}
		self.freq_pref = {}
		self.final_pref = {}

		self.words = []
		self.wordpairs = []

	@classmethod
	def set_stopwords(cls,stopwordfile):
		try:
			with open(stopwordfile) as f:
				for line in f:
					cls.stop_words.append(line.strip().decode('utf-8'))

		except Exception,e:
			print e.message

		return True

	@classmethod
	def is_retain(cls,term,pos):
		if len(term)>1 and term not in cls.stop_words and ( pos in cls.pos_legal  or (pos=="eng" and len(term)>2 and term[0].isupper())):
			return True
		else:
			return False

	def calc_wordpairs(self,newsdict):
		first = newsdict.get('FIRST_SENTENCE',"")
		last = newsdict.get('LAST_SENTENCE',"")
		mid = newsdict.get('MID_SENTENCE',"")
		whole_text = first + mid + last
		if not whole_text:
			print 'news is none!\n'
			return False

		title_str = ""
		if newsdict.has_key('TITLE') and newsdict['TITLE']:
			title = []
			for pair in newsdict['TITLE'].split(' '):
				if len(pair.split("/")) == 2:
					term,pos = pair.split("/")
					title.append(term)
			title_str = "".join(title)

		for pair in whole_text.split(' '):
			if len(pair.split("/")) == 2:
				term,pos = pair.split("/")
				term_info = {}
				if TextRank.is_retain(term, pos):
					term_info['pos'] = pos
					if first and term in first:
						term_info['first'] = True
					else:
						term_info['first'] = False
					if last and term in last:
						term_info['last'] = True
					else:
						term_info['last'] = False
					if title_str and term in title_str:
						term_info['title'] = True
					else:
						term_info['title'] = False
					self.wordpairs.append((term,term_info))
					self.words.append(term)
		return True

	def calc_preference(self,method=0):
		if method == 0:
			for term,info in self.wordpairs:
				# calc pos preference
				if self.pos_pref.get(term) == None:
					self.pos_pref[term] = []
				self.pos_pref[term].append(self._get_pos_preference(info['pos']))
				# calc loc preference
				if self.loc_pref.get(term) == None:
					self.loc_pref[term] = self._get_loc_preference_method0(term,info['title'],info['first'],info['last'])

		elif method == 1:
			for term,info in self.wordpairs:
				# calc pos preference
				if self.pos_pref.get(term) == None:
					self.pos_pref[term] = []
				self.pos_pref[term].append(self._get_pos_preference(info['pos']))
				# calc loc preference
				if self.loc_pref.get(term) == None:
					self.loc_pref[term] = self._get_loc_preference_method1(term,info['title'],info['first'],info['last'])
				# calc freq preference
				if self.freq_pref.get(term) == None:
					self.freq_pref[term] = 0
					self.freq_pref[term] += 1

					'''
					if self.freq_pref.get(term) == None:
						self.freq_pref[term] = 0
					self.freq_pref[term] += 1
					'''
		'''
		for term in self.freq_pref.keys():
			self.freq_pref[term] = self.freq_pref[term] / len(self.words)
		'''
		if self.pos_pref:
			for w in list(set(self.words)):
				self.pos_pref[w] = np.array(self.pos_pref[w]).mean()

		#return self.final_pref
		if self.pos_pref:
			self.final_pref['pos'] = self.pos_pref
		if self.loc_pref:
			self.final_pref['loc'] = self.loc_pref
		if self.freq_pref:
			self.final_pref['freq'] = self.freq_pref

	def _get_pos_preference(self,pos):
		#根据不同的词性给予词汇不同的分数，分数来源参考"基于语义的中文文本关键词提取算法"
		s=0.1
		if pos[0]=="n":
			s=0.8
		elif pos=="j":
			s=0.7
		elif pos in ("an","i","I","vn"):
			s=0.6
		elif pos in ("eng","a"):
			s=0.5
		return s

	def _get_loc_preference_method1(self,term,title,first_sentences,last_sentences):
		# 根据是否在标题中出现，给一个权重
		i = 1.0
		if title:
			i = LAMBDA_TITLE
		if first_sentences:
			i = LAMBDA_FIRST
		if last_sentences:
			i = LAMBDA_LAST
		return i


	def _get_loc_preference_method0(self,term,title,first_sentences,last_sentences):
		#根据词汇出现的位置打分（标题，（段）首句，（段）未句）
		s=0.0
		if title:
			s += 0.8
		if first_sentences:
			s += 0.1
		if last_sentences:
			s += 0.1
		return s

	def textrank(self, newsdict=None,topK=10,method=0):
		# final rank
		if not newsdict:
			raise ValueError('words not given')

		#final_pref = self._calc_preference(wordpair)

		g = UndirectWeightedGraph(method)
		cm = defaultdict(int)
		for i, w in enumerate(self.words):
			for j in xrange(i + 1, i + self.span):
				if j >= len(self.words):
					break
				cm[(w, self.words[j])] += 1
		for terms, w in cm.items():
			g.addEdge(terms[0], terms[1], w)

		#print "words:"," ".join(self.words)
		#print "nodes:", " ".join(g.graph.keys())

		nodes_rank = g.rank(self.final_pref,method)

		tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
		if len(tags)>topK:
			return tags[:topK]
		else:
			return tags


class w2v_extract():

	def __init__(self,tmpwordsfile):
		self.model = None
		self.all_words_file = tmpwordsfile

	def pre_train(self,doc_gen):
		with codecs.open(self.all_words_file,'w','utf-8') as fw:
			for newsdict in doc_gen:
				first = newsdict.get('FIRST_SENTENCE',"")
				last = newsdict.get('LAST_SENTENCE',"")
				mid = newsdict.get('MID_SENTENCE',"")
				title = newsdict.get('TITLE',"")
				whole_text = title + first + mid + last
				if not whole_text:
					print 'news is none!\n'
					continue

				for pair in whole_text.split(' '):
					if len(pair.split("/")) == 2:
						term,pos = pair.split("/")
						if TextRank.is_retain(term, pos):
							fw.write(term + ' ')
				fw.write('\n')

	def train_model(self,modelpath,retrain=True):
		if not retrain:
			self.model = w2v.Word2Vec.load(modelpath)
		else:
			# must use skip gram with softmax
			self.model = w2v.Word2Vec(w2v.LineSentence(self.all_words_file),sg=1,hs=1,negative=0,size=200,window=5,min_count=5,workers=cpu_count())
			self.model.save(modelpath)

		return True

	def _predict_proba(self,oword, iword):
		iword_vec = self.model[iword]
		oword = self.model.wv.vocab[oword]
		oword_l = self.model.syn1[oword.point].T
		dot = np.dot(iword_vec, oword_l)
		lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
		return lprob

	def keywords_dict(self,newsdict,topK=10):
		wordlist = []
		first = newsdict.get('FIRST_SENTENCE',"")
		last = newsdict.get('LAST_SENTENCE',"")
		mid = newsdict.get('MID_SENTENCE',"")
		title = newsdict.get('TITLE',"")
		whole_text = title + first + mid + last
		if not whole_text:
			print 'news is none!\n'
			return None

		for pair in whole_text.split(' '):
			if len(pair.split("/")) == 2:
				term,pos = pair.split("/")
				if TextRank.is_retain(term, pos):
					wordlist.append(term)

		s = [w for w in wordlist if w in self.model]
		ws = {w: sum([self._predict_proba(u, w) for u in s]) for w in s}

		kw_pairs = Counter(ws).most_common(n=topK)
		return [w for w,v in kw_pairs]


def preprocess(srcdir):
	if srcdir:
		p = Path(srcdir)
		# self.total_count = len(list(p.glob("**/*.json")))
		for jsfile in p.glob("**/*.json"):
			with open(str(jsfile), 'r') as fr:
					text = fr.read()
					yield json.loads(text, strict=False)
	return

def dycj_sqlite_preprocess(dbdict,splitter):
	if dbdict and ds:
		for db,tablist in dbdict.items():
			try:
				for tabname in tablist:
					conn = sqlite3.connect(db)
					print 'db %s connect ok' % db
					c = conn.cursor()
					stmt = '''
						select title,content from %s where 
						publish_date > datetime('2017-01-01') 
						and publish_date < datetime('2017-12-31');
						''' % tabname
					c.execute(stmt)
					for row in c:
						news_dict = {}
						news_dict['TITLE'] = splitter.split_one_string(row[0])
						news_dict['MID_SENTENCE'] = splitter.split_one_string(row[1])
						yield news_dict

			except Exception,e:
				print e.message
				continue
			finally:
				conn.close()

	return None

def evaluate(newsdict,keywords,target_keywords):
	first = newsdict.get('FIRST_SENTENCE',"")
	last = newsdict.get('LAST_SENTENCE',"")
	mid = newsdict.get('MID_SENTENCE',"")

	N = len(mid) + len(first) + len(last)

	keywords_list = None
	if '|' in target_keywords:
		keywords_list = target_keywords.split('|')
	else:
		keywords_list = target_keywords

	tp = fp = tn = fn = 0.0
	for kw in keywords:
		if kw in keywords_list:
			tp += 1
		else:
			fp += 1


	tn = N - len(keywords_list) - fp
	fn = len(keywords_list) - tp

	pre = tp / (tp + fp)
	rec = tp / (tp + fn)

	if 0.0 < pre < 0.01 or 0.0 < rec < 0.1:
		print 'tp is %d,tp+fn is %d\n' % (tp,tp+fn)

	f = pre*rec*2 / (pre + rec + 0.01)

	return (pre,rec,f)


def calc_H_pref(all_dicts):
	N = len(all_dicts)
	all_words = {}
	all_words_in_news = {}
	current_news_term_count = {}
	for news in all_dicts:
		if not news['first']:
			first = ""
		else:
			first = news['first']
		if not newsdict['last']:
			last = ""
		else:
			last = news['last']

		all_text = first + news['content'] + last
		all_words_in_news[news['id']] = {}

		for pair in all_text.split(' '):
			if len(pair.split("/")) == 2:
				term,pos = pair.split("/")
				if TextRank.is_retain(term, pos):
					if current_news_term_count.get(term) == None:
						current_news_term_count[term] = 0
					current_news_term_count[term] += 1
					if all_words.get(term) == None:
						all_words[term] = 0
					all_words[term] += 1

		for term,count in current_news_term_count.items():
			all_words_in_news[news['id']][term] = count

	pref = defaultdict(float)

	for term in all_words.keys():
		s = 0.0
		for k in all_words_in_news.keys():
			s += all_words_in_news[k][term] / all_words[term] * math.log( all_words[term] / all_words_in_news[k][term])
		pref[term] = 1.0 / math.log(N+0.1) * s

	return pref

def calc_eval_result(eval_list):
	sump = sumr = sumf = 0.0
	count = len(eval_list)

	for result in eval_list:
		sump += result[0]
		sumr += result[1]
		sumf += result[2]

	return (sump/count , sumr/count , sumf/count)


if __name__=="__main__":
	text="\u4e0a\u5e02\u516c\u53f8/nt  \u6b63/d  \u8d1f\u9762/n  \u6d88\u606f/n  \u901f\u89c8/n  \uff1a/x  \u8fd1/t  10/m  \u5bb6/m  \u516c\u53f8/n  \u80a1\u4e1c/n  \u589e\u6301/v  \uff0c/x  \u5929\u6daf\u793e\u533a/l  \u5c06/d  \u6b63\u5f0f/ad  \u6302\u724c/v  \n\n\u3000/x  \u3000/x  \u9999\u6e2f/ns  \u4e07/m  \u5f97/ud  \u901a\u8baf\u793e/nt  \u6574\u7406/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u2014/x  \u2014/x  \u2014/x  \u2014/x  \u6b63\u9762/ad  \u6d88\u606f/n  \u2014/x  \u2014/x  \u2014/x  \u2014/x  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6d69\u4e30\u79d1\u6280/nt  \u5b9a/v  \u589e/v  \u4f5c\u4ef7/n  7.45/m  \u4ebf/m  \u5e76\u8d2d/v  \u8def\u5b89/ns  \u4e16\u7eaa/nt  </x  br/eng  >/x  \u3000/x  \u3000/x  \u83b1\u8335\u7f6e\u4e1a/nt  \u63a8/v  1270/m  \u4e07\u4efd/m  \u80a1\u6743/n  \u6fc0\u52b1/v  \u8ba1\u5212/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u9ed1\u829d\u9ebb/nt  \u5b9a/v  \u589e\u52df/v  \u8d44/n  16/m  \u4ebf/m   /x  \u52a0\u7801/n  \u4e3b\u4e1a/n  \u6d89/v  \u4e92\u8054\u7f51/n  +/x  </x  br/eng  >/x  \u3000/x  \u3000/x  \u9752\u677e\u80a1\u4efd/nt  \u63a8/v  6.5/m  \u4ebf\u5143/m  \u5b9a/v  \u589e/v  \u9884\u6848/n   /x  8/m  \u6708/m  3/m  \u65e5/m  \u590d\u724c/v  </x  br/eng  >/x  \u3000/x  \u3000/x  \u7ef4\u5c14\u5229/nt  \u4e2d\u671f/t  \u51c0\u5229/n  \u589e\u8fd1/v  9/m  \u6210/n   /x  \u8bbe/v  \u5362\u68ee\u5821/ns  \u5b50\u516c\u53f8/n  \u5e03\u5c40/n  \u6d77\u5916/ns  \u5e02\u573a/ns  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6052\u987a\u4f17\u6607/nt  \u4e1a\u7ee9/n  \u5927\u589e/v  \u62df/v  10/m  \u8f6c/v  10/m  \u9001/v  5/m  \u6d3e/v  0.5/m   /x  \u5728/p  \u975e/d  \u8bbe/v  \u4e24/m  " \
	"\u5b50\u516c\u53f8/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u4fdd\u9f84\u5b9d/nt  \u7b79\u5212/n  \u6536\u8d2d/v  \u6559\u80b2/ns  \u884c\u4e1a/n  \u516c\u53f8/n   /x  8/m  \u6708/m  3/m  \u65e5/m  \u8d77/v  \u505c\u724c/v  </x  br/eng  >/x  \u3000/x  \u3000/x  \u7d2b\u5149\u80a1\u4efd/nt  \u518d/d  \u83b7/v  \u7d2b\u5149/nr  \u96c6\u56e2/ns  \u53ca/c  \u4e00\u81f4/d  \u884c\u52a8/vn  \u4eba/n  \u65a5\u8d44/v  \u903e/vg  \u5343\u4e07\u5143/m  \u589e\u6301/v  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5343\u5c71\u836f\u673a/nt  \u6df1\u5ea6/ns  \u5e03\u5c40/n  \u201c/x  \u7cbe\u51c6/n  \u533b\u7597/n  \u201d/x   /x  \u63a7\u80a1/v  \u80a1\u4e1c/n  \u589e\u6301/v  \u5f70\u663e/v  \u4fe1\u5fc3/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u4e1c\u6e90\u7535\u5668/nt  \u83b7/v  \u5b9e\u9645/n  \u63a7\u5236/v  \u4eba/n  \u7d2f\u8ba1/v  \u589e\u6301/v  76.5/m  \u4e07\u80a1/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u4f73\u9686\u80a1\u4efd/nt  \u83b7/v  \u5b9e\u9645/n  \u63a7\u5236/v  \u4eba/n  \u589e\u6301/v  388/m  \u4e07\u80a1/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6c38\u5927\u96c6\u56e2/nt  \u62df/v  2/m  \u4ebf\u5143/m  \u524d\u6d77/ns  \u8bbe/v  \u6295\u8d44/vn  \u516c\u53f8/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6842\u4e1c\u7535\u529b/nt  \u5168\u8d44/n  \u5b50\u516c\u53f8/n  \u589e\u6301/v  \u56fd\u6d77\u8bc1\u5238/nt  20/m  \u4e07\u80a1/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u91cd\u9633\u6295\u8d44/nt  \u4e3e\u724c/v  \u4e0a\u6d77\u5bb6\u5316/nt  </x  br/eng  >/x  \u3000/x  \u3000/x  \u987a\u8363\u4e09\u4e03/nt  \u5b50\u516c\u53f8/n  \u4e0e/p  \u661f\u7693/i  \u5f71\u4e1a/vn  \u5408\u4f5c/ns   /x  \u8fdb\u519b/v  " \
	"\u5f71\u89c6/n  \u884c\u4e1a/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6c5f\u4e2d\u836f\u4e1a/nt  \u4e2d\u671f/t  \u51c0\u5229/n  1.39/m  \u4ebf/m   /x  \u540c\u6bd4/j  \u589e\u957f/v  45/m  %/x  </x  br/eng  >/x  \u3000/x  \u3000/x  \u529b\u6e90\u4fe1\u606f/nt  \u4e0a\u534a\u5e74/t  \u51c0\u5229/n  \u540c\u6bd4/j  \u5927\u589e/v  \u903e/vg  \u4e00\u500d/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u65b0\u5357\u6d0b/nt  \u4e94/m  \u9ad8\u7ba1/n  \u589e\u6301/v  5/m  \u4e07\u80a1/m   /x  \u6b63/d  \u7814\u7a76/vn  \u80a1\u6743/n  \u6fc0\u52b1/v  \u8ba1\u5212/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5929\u55bb\u4fe1\u606f/nt  \u83b7/v  \u63a7\u80a1/v  \u80a1\u4e1c/n  \u589e\u6301/v  107/m  \u4e07\u80a1/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5965\u514b\u80a1\u4efd/nt  1.3/m  \u4ebf/m  \u8d2d/vg  \u4e1c\u7855/nr  \u73af\u4fdd/j  37/m  %/x  \u80a1\u6743/n   /x  \u63a8\u52a8/v  \u7eff\u8272/n  \u73af\u4fdd/j  \u4e1a\u52a1/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5927\u4e1c\u65b9/nt  \u7ea6/d  1900/m  \u4e07/m  \u6536\u8d2d/v  \u5b50\u516c\u53f8/n  \u767e\u4e1a/n  \u8d85\u5e02/v  40/m  %/x  \u80a1\u6743/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6c38\u9f0e\u80a1\u4efd/nt  \u63a8/v  \u5458\u5de5/n  \u6301\u80a1/v  \u8ba1\u5212/n   /x  \u5c06/d  \u8d2d\u5165/v  \u7ea6/d  300/m  \u4e07\u80a1/m  \u516c\u53f8\u80a1\u7968/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u795e\u5dde\u6cf0\u5cb3/nt  \u589e\u8d44/v  \u63a7\u80a1/v  \u5b50\u516c\u53f8/n   /x  \u52a0\u7801/n  \u79fb\u52a8/vn  \u4e92\u8054\u7f51/n  \u5927/a  \u6570\u636e/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u534e\u5a92\u63a7\u80a1/nt  \u4e0e/p  \u963f\u91cc/ns  \u6218\u7565/n  " \
	"\u5408\u4f5c/ns   /x  \u6253\u9020/v  \u57ce\u5e02/ns  \u751f\u6d3b/vn  \u670d\u52a1\u5546/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6676\u76db\u673a\u7535/nt  \u5b9a/v  \u589e\u52df/v  \u8d44/n  16/m  \u4ebf\u5143/m   /x  \u52a0\u7801/n  \u84dd\u5b9d\u77f3/nr  \u9879\u76ee/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u94c1\u6c49\u751f\u6001/nt  9600/m  \u4e07/m  \u6536\u8d2d/v  \u73af\u53d1/j  \u73af\u4fdd/j  80/m  %/x  \u80a1\u6743/n   /x  \u52a0\u7801/n  \u73af\u4fdd/j  \u9886\u57df/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u7231\u5c14\u773c\u79d1/nt  \u65a5\u8d44/v  2/m  \u4ebf/m  \u518d\u6b21/d  \u53c2\u4e0e/v  \u5e76\u8d2d/v  \u57fa\u91d1/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6210\u90fd\u8def\u6865/nt  \u4e2d\u6807/ns  1.81/m  \u4ebf\u5143/m  PPP/eng  \u9879\u76ee/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6d3d\u6d3d\u98df\u54c1/nt  \u63a8/v  1.5/m  \u4ebf\u5143/m  \u5458\u5de5/n  \u6301\u80a1/v  \u8ba1\u5212/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u68ee\u8fdc\u80a1\u4efd/nt  \u8bbe/v  \u5408\u8d44/vn  \u516c\u53f8/n   /x  \u4fc3\u8fdb/ns  \u5927\u578b/b  \u518d\u751f/v  \u8bbe\u5907/vn  \u63a8\u5e7f\u5e94\u7528/i  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6da6\u90a6\u80a1\u4efd/nt  \u4e0e/p  \u6e05\u63a7/vn  \u8d44\u7ba1/n  \u5408\u4f5c/ns   /x  \u63a8\u52a8/v  \u8282\u80fd/v  \u73af\u4fdd/j  \u4ea7\u4e1a/n  \u53d1\u5c55/vn  </x  br/eng  >/x  \u3000/x  \u3000/x  \u65af\u7c73\u514b/nt  \u62df/v  \u6218\u7565/n  \u5165\u80a1/v  \u946b/nr  \u5c71/n  \u4fdd\u9669\u4ee3\u7406/n  \u516c\u53f8/n  </x  br/eng  >/x  \u3000/x  \u3000/x  */x  ST/eng  \u4e50/a  \u7535/n  \u4e2d\u671f/t  \u51c0\u5229/n  \u540c\u6bd4/j  \u589e/v  " \
	"\u4e09\u6210/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u798f\u751f\u4f73/nz  \u4fe1/n  225/m  \u4e07/m  \u53c2\u80a1/v  \u8bbe\u7acb/v  \u4e0a\u95e8/ns  \u63a8\u62ff/v  \u670d\u52a1\u516c\u53f8/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5206\u8c46/n  \u6559\u80b2/ns  \u5b9a/v  \u589e\u52df/v  \u8d44/n  4.5/m  \u4ebf/m  \u52a0\u7801/n  \u4e3b\u4e1a/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u5929\u6daf\u793e\u533a/l  \u5c06/d  \u6b63\u5f0f/ad  \u6302\u724c/v   /x  \u52df/vg  \u8d44/n  3999/m  \u4e07\u5143/m  \u89e3\u51b3/v  \u77ed\u671f/b  \u6d41\u52a8\u8d44\u91d1/n  \u9700\u6c42/v  </x  br/eng  >/x  \u3000/x  \u3000/x  \u2014/x  \u2014/x  \u2014/x  \u2014/x  \u8d1f\u9762/n  \u6d88\u606f/n  \u2014/x  \u2014/x  \u2014/x  \u2014/x  </x  br/eng  >/x  \u3000/x  \u3000/x  \u4e07\u8fbe\u9662\u7ebf/nt  \u6f84\u6e05/v  \u738b\u5065\u6797/nr  \u59bb\u5b50/n  \u5185\u5e55/n  \u4ea4\u6613/n  \u4f20\u95fb/n  </x  br/eng  >/x  \u3000/x  \u3000/x  \u795e\u5f00\u80a1\u4efd/nt  \u603b\u7ecf\u7406/n  \u6d89/v  \u77ed\u7ebf\u4ea4\u6613/n  \u88ab/p  \u8bc1\u76d1\u4f1a/j  \u7acb\u6848/n  \u8c03\u67e5/vn  </x  br/eng  >/x  \u3000/x  \u3000/x  \u4e2d\u6d77\u8fbe/nt  \u4e0a\u534a\u5e74/t  \u51c0\u5229/n  \u4e0b\u964d/v  \u903e/vg  \u4e03\u6210/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u7696\u6c5f\u7269\u6d41/nt  \u6536\u5230/v  \u8bc1\u76d1\u4f1a/j  \u884c\u653f\u5904\u7f5a/n  \u4e66/n   /x  \u88ab/p  \u8b66\u544a/n  \u5e76/c  \u7f5a\u6b3e/n  50/m  \u4e07\u5143/m  </x  br/eng  >/x  \u3000/x  \u3000/x  \u6d77\u5b81\u76ae\u57ce/nt  \u4e0a\u534a\u5e74/t  \u51c0\u5229/n  5.11/m  \u4ebf/m   /x  \u540c\u6bd4/j  \u964d/v  14/m  %/x  </x  br/eng  >/x  \u3000/x  " \
	"\u3000/x  \u90e8\u5206/n  \u4ea7\u54c1/n  \u5904/n  \u57f9\u80b2/vn  \u671f/n   /x  \u5343\u91d1\u836f\u4e1a/nt  \u4e2d\u671f/t  \u51c0\u5229/n  \u4e0b\u6ed1/v  8.5/m  %/x  </x  br/eng  >/x  \u3000/x  \u3000/x"
	# news
	# id :
	# title :
	# first :
	# last :
	# content :
	# keywords :

	'''
	newsdict = {}
	newsdict['OBJECT_ID'] = 1
	newsdict['TITLE'] = None
	newsdict['FIRST_SENTENCE'] = None
	newsdict['MID_SENTENCE'] = None
	newsdict['LAST_SENTENCE'] = text
	newsdict['KEYWORDS'] = ['净利','股权','股东','基本面','胡扯']
	#u'TITLE', u'URL', u'LAST_SENTENCE', u'OBJECT_ID', u'FIRST_SENTENCE', u'SOURCE', u'MID_SENTENCE', u'DATE', u'KEYWORDS', u'SECTIONS'
	'''
	ds = split_word.doc_splitter('/home/lc/ht_work/ML/old_txt','/home/lc/ht_work/ML/new_txt/allwords.txt','/home/lc/ht_word/xwparse/stopwords_merge.txt','/home/lc/ht_work/xwparse/userdict.txt',True)

	approach = 0
	topK = 5
	process_count = 0
	eval_result = []

	#for kmeans
	cluster_num = 100
	iter_max = 30
	prt_num = 2

	TextRank.set_stopwords('/home/lc/ht_work/news_textrank/stopwords_wz.txt')
	if approach == 2:
		UndirectWeightedGraph.set_model(MODEL_PATH)

	newsdict_gen = preprocess(CORPORA_PATH)

	# construct dycj db info
	dycj_dbs = {}
	dycj_dbs['']


	newsdict_gen_dycj = dycj_sqlite_preprocess(dbdict=dycj_dbs,splitter=ds)

	keywords_list = []

	if approach == 3:
		w2v_new = w2v_extract(TMPWORDFILE)
		#w2v_new.pre_train(js_gen)
		w2v_new.train_model(MODELPATH_3,retrain=False)
		newsdict_gen_1 = preprocess(CORPORA_PATH)

		for newsdict in newsdict_gen_1:
			try:
				keywords = w2v_new.keywords_dict(newsdict,topK=topK)
				print " ".join(keywords)

				if newsdict.has_key('KEYWORDS') and newsdict['KEYWORDS']:
					res = evaluate(newsdict,keywords,newsdict['KEYWORDS'])
					eval_result.append(res)
					print "precision : %f \t recall : %f \t f : %f\n" % (res[0],res[1],res[2])

				process_count += 1
			except Exception as e:
				print e.message
				continue
	else:
		f = codecs.open('all_news_keywords.txt','w',encoding='utf-8')
		for newsdict in newsdict_gen_dycj:
			try:
				tr=TextRank()
				ret = tr.calc_wordpairs(newsdict)
				if not ret:
					continue
				tr.calc_preference(approach)

				'''
				if approach == 1:
					H_pref = calc_H_pref(all_dicts)
					tr.set_freq_pref(H_pref)
				'''
				keywords = tr.textrank(newsdict,topK=topK,method=approach)
				if keywords:
					keywords_list.append(keywords)
					#print " ".join(keywords)
					#f.write(" ".join(keywords))
					#f.write('\n')

				'''
				if newsdict.has_key('KEYWORDS') and newsdict['KEYWORDS']:
					res = evaluate(newsdict,keywords,newsdict['KEYWORDS'])
					eval_result.append(res)
					print "precision : %f \t recall : %f \t f : %f\n" % (res[0],res[1],res[2])
				'''

				process_count += 1
			except Exception as e:
				print e.message
				continue
		f.close()

	#all keywords in keywords_list now

	'''
	kmeans_obj = vector_clustering.kmean_clustering('/home/lc/ht_work/data/200_0_0_1/news.model',keywords_list,cluster_num,iter_max)
	cents,clusterAssment = kmeans_obj.clustering()
	kmeans_obj.print_cluster_nearest_words('/home/lc/ht_work/data/w2v_cluster.txt',clusterAssment,prt_num)
	'''
	'''
	kmeans_obj = vector_clustering.kmean_sklearn_clustering('/home/lc/ht_work/data/200_0_0_1/news.model',keywords_list,step=10)
	kmeans_obj.clustering(115)
	'''




	'''
	h_obj = vector_clustering.hierarchy_clustering('/home/lc/ht_work/data/200_0_0_1/news.model',keywords_list)
	# through small samples test : 0.8 is best .值越大，类内含义范围越广泛
	cluster_labels = h_obj.clustering(0.75)
	#h_obj.print_cluster_words('/home/lc/ht_work/news_textrank/hcluster_res.txt',cluster_labels,10)
	'''

	'''
	print "process %Ld files , %Ld files with keywords to evaluate\n" % (process_count,len(eval_result))
	final_res = calc_eval_result(eval_result)
	print "final result ==> precision : %f \t recall : %f \t f : %f\n" % (final_res[0],final_res[1],final_res[2])
	'''


