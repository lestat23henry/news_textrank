# coding=utf-8
# author : liangchen
# decription : class doc_splitter to split all words in one directory path,generate new files(with segmented words in)

import os
from multiprocessing import cpu_count
import re
from datetime import datetime

from pathlib import Path
import jieba
import jieba.posseg as pseg

class doc_splitter():
	def __init__(self,srcdir,targetfile,stopword=None,userdict=None,parallel=True):
		self.srcdir = srcdir
		self.tagfile = targetfile

		self.stopword = stopword  #dict
		self.parallel = parallel
		self.userdict = userdict

		jieba.initialize()
		if self.userdict:
			jieba.load_userdict(userdict)

		if self.parallel:
			jieba.enable_parallel(cpu_count())

	def utf8_one_doc(self,filepath):
		if not filepath:
			return

		print u'time: %s ==> 转换文件%s为utf-8格式\n' % (datetime.now(),filepath)
		filepath_utf8 = '/tmp' + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + '_utf8' + '.txt'

		if os.path.exists(filepath_utf8):
			print u'time: %s ==> 文件%s已经转换为%s\n' % (datetime.now(), filepath,filepath_utf8)
			return filepath_utf8

		with open(str(filepath),'r') as fr:
			with open(filepath_utf8,'w') as fw:
				line = fr.readline()
				while line:
					if line == '\r\n':
						line = fr.readline()
						continue
					newline = line.decode('GB18030').encode('utf-8')
					print newline
					print >> fw, newline
					line = fr.readline()

		print u'time: %s ==> 文件%s已经转换为%s\n' % (datetime.now(), filepath, filepath_utf8)
		return filepath_utf8

	def split_one_string(self,str):
		if not str:
			return None
		try:
			outstr = u""
			line_str = str.strip()
			line1 = re.sub("[\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+－——！，;:：。？、~@#￥%……&*（）]+".decode("utf8"),
								   "".decode("utf8"), line_str)

			#word_list = list(jieba.cut(line1,cut_all=False,HMM=True))  # 用结巴分词，对每行内容进行分词
			word_pair_list = pseg.cut(line1)
			if self.stopword:
				out_pair = [wordpair for wordpair in word_pair_list if wordpair.word not in self.stopword]
			else:
				out_pair = word_pair_list

			for pair in out_pair:
				outstr += pair.word + "/" + pair.flag + " "

		except Exception,e:
			print e.message

		return outstr


	def split_one_doc(self,filepath):
		if not filepath:
			return

		print u'time: %s ==> 对文件%s进行分词\n' % (datetime.now(), filepath)
		#filepath_segmented = self.tagdir + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + '_segmented' + '.txt'
		filepath_segmented = self.tagfile

		with open(filepath,'r') as fr:
			with open(filepath_segmented,'a') as fw:
				line = fr.readline()
				while line:
					if line == '\r\n':
						line = fr.readline()
						continue
					line_str = line.strip().decode('utf-8', 'ignore')  # 去除每行首尾可能出现的空格，并转为Unicode进行处理
					line1 = re.sub("[\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+－——！，;:：。？、~@#￥%……&*（）]+".decode("utf8"),
								   "".decode("utf8"), line_str)

					word_list = list(jieba.cut(line1,cut_all=False,HMM=True))  # 用结巴分词，对每行内容进行分词
					if self.stopword:
						out_str = [word+' ' for word in word_list if word not in self.stopword]
					else:
						out_str = word_list
					fw.write(" ".join(out_str).strip().encode('utf-8') + '\n')  # 将分词好的结果写入到输出文件
					line = fr.readline()

		print u'time: %s ==> 文件%s分词结束，保存到%s\n' % (datetime.now(), filepath, filepath_segmented)
		return filepath_segmented


	def split_all(self):
		p = Path(self.srcdir)
		for txt in p.glob("**/*.txt"):
			txt_utf8 = self.utf8_one_doc(str(txt))
			self.split_one_doc(txt_utf8)

		return True


#class test:
if __name__=='__main__':
	ds = doc_splitter('/home/lc/ht_work/ML/old_txt','/home/lc/ht_work/ML/new_txt/allwords.txt','/home/lc/ht_word/xwparse/stopwords_merge.txt','/home/lc/ht_work/xwparse/userdict.txt',True)
	wordpair = ds.split_one_string(u'小明硕士毕业于中国科学院计算所，后在日本京都大学深造')
	#ds.split_all()

