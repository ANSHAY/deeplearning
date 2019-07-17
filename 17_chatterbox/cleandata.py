import os

#data_path = 'D:\\Datasets\\chatterbot-corpus-master\\chatterbot_corpus\\data\\english\\'
#files = os.listdir(data_path)
data_path = ''
files = ['train2.txt']

corpus = ''

for file in files:
    with open(data_path+file) as f:
        data = f.read()
        corpus += data
        f.close()

corpus = corpus.replace('\n\n','\n')
corpus = corpus.replace('A: ','')
corpus = corpus.replace('B: ','')

with open('conversation_train2.txt', 'w') as f:
    f.write(corpus)
    f.close()
print('done')
