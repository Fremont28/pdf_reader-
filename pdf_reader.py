import re 
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from email.generator import Generator
import nltk 

#pdf to txt 
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

file1=convert_pdf_to_txt('sample.pdf')

f=open('xxx.txt','w')
f.write(file1)
f.close() 

with open('xxx.txt') as f:
    clean_pdf=f.read().splitlines() 
clean_pdf

#clean pdf 
shear=[i.replace('\xe2\x80\x9c','') for i in clean_cont]
shear=[i.replace('\xe2\x80\x9d','') for i in shear ]
shear=[i.replace('\xe2\x80\x99s','') for i in shear ]

shears = [x for x in shear if x != ' ']
shearss = [x for x in shears if x != '']

dubby=[re.sub("[^a-zA-Z]+"," ",s) for s in shearss]

from nltk.tokenize import word_tokenize 
import re 
import regex
import string 
from nltk.corpus import stopwords 

# apply word_tokenize to each element of the list 
dubby1=[word_tokenize(x) for x in dubby]

#remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
tokenized_text=[]
for review in dubby1:
    new_text = []
    for token in review: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_text.append(new_token)

    tokenized_text.append(new_text)
tokenized_text

#remove filler words
tokenized_no_stop=[]
for report in tokenized_text:
    new_text1=[]
    for word in report:
        if not word in stopwords.words('english'):
            new_text1.append(word)
    tokenized_no_stop.append(new_text1)
tokenized_no_stop 

ts_string= ''.join(str(e) for e in tokenized_no_stop)
ts_string1=ts_string.split(' ')

#topic modelling 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# Counting the no of times each word appears in the text 
vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
dtm=vect.fit_transform(ts_string1)
pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())

#topic modelling 
lda=LatentDirichletAllocation(n_components=5)
lda_dft=lda.fit_transform(dtm)

#sort topics 
sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vect.get_feature_names())

import mglearn
mglearn.tools.print_topics(topics=range(5), feature_names=features,
sorting=sorting, topics_per_chunk=5, n_words=10)

