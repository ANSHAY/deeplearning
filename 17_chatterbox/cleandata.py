import os

#data_path = 'D:\\Datasets\\chatterbot-corpus-master\\chatterbot_corpus\\data\\english\\'
#files = os.listdir(data_path)
data_path = ''
files = ['conversation_train3.txt']

corpus = ''

for file in files:
    with open(data_path+file) as f:
        data = f.read()
        corpus += data
        f.close()

corpus = corpus.replace('            Search Images      Translate','')
corpus = corpus.replace('1\n Repeat','')
corpus = corpus.replace('2\n Repeat','')
corpus = corpus.replace('3\n Repeat','')
corpus = corpus.replace('4\n Repeat','')
corpus = corpus.replace('5\n Repeat','')
corpus = corpus.replace('6\n Repeat','')
corpus = corpus.replace('Practice the Conversations of This Topic with Mike','')
corpus = corpus.replace('\nBack\n','')
corpus = corpus.replace('Copyright © 2019. All rights reserved.','')
corpus = corpus.replace('eslfast.com','')
corpus = corpus.replace('? ','\n')
corpus = corpus.replace('. ','\n')
corpus = corpus.replace(', ','\n')
corpus = corpus.replace('! ','\n')
corpus = corpus.replace('\n\n','\n')
corpus = corpus.replace('\n\n','\n')
corpus = corpus.replace('\n\n','\n')

corpus = corpus.replace('A: ','')
corpus = corpus.replace('B: ','')

with open('conversation_train3.txt', 'w') as f:
    f.write(corpus)
    f.close()
print('done')
