source_file_path = "../data/chats.txt"
SPLIT_SIZE = 5000 # integer, number of lines per file
output_file_path = "../data/chats/"
corpus = []

## read source file line by line
with open(source_file_path, 'r', encoding='utf-8') as fin:
    corpus = fin.readlines()

LEN = len(corpus)
itr = LEN//SPLIT_SIZE + 1

for i in range(itr):
    with open(output_file_path+"chat_"+str(i)+".txt", 'w+', encoding='utf-8') as fo:
        for j in range(SPLIT_SIZE):
            n = i*SPLIT_SIZE+j
            if n>=LEN:
                break
            fo.write(corpus[n])
