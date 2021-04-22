import re
from sklearn.datasets import load_files
train_data = load_files('DL_dataset/train',encoding='utf-16')
test_data = load_files('DL_dataset/test',encoding='utf-16')
with open('dictionary/words.txt',encoding='utf-8') as dictionary_file:
    dictionary = dictionary_file.read().replace('\n',' ')

def create_stopwordlist():
    f = open('dictionary/vietnamese-stopwords.txt', encoding='utf-8')
    data = []
    null_data = []
    for i, line in enumerate(f):
        line = repr(line)
        line = line[1:len(line)-3]
        data.append(line)
    return data
stopwords_vn = create_stopwordlist()

def remove_html(text):
    # Bỏ tag html (nếu có)
    return re.sub(r'<[^>]*>', '', text) 
def remove_numbers(text):
    # Xóa chữ số
    return re.sub(r'\d+',' ',text)
def remove_punctuations(text):
    return re.sub(r'[^\w\s]', ' ', text)
def lowercase_text(text):
    text =  text.lower()
    return re.sub(r'\s+', ' ', text).strip()
def remove_invalid_words(text):
    words_in_text  = text.split()
    text =  [word for word in words_in_text if (word in dictionary and word not in stopwords_vn)]
    return ' '.join(word for word in text)
def label_add(text,label):
    label = re.sub(' ','_',label.lower())
    text_with_label = '__label__'+label+' '+text
    return text_with_label

    
def document_preprocessing(document,label):
    document = remove_html(document)
    document = remove_numbers(document)
    document = remove_punctuations(document)
    document = lowercase_text(document)
    document = remove_invalid_words(document)
    document = label_add(document,label)
    return document

with open('preprocess_data/train_data.txt','w') as train_data_file:
    for i in range(len(train_data.data)):
      train_data_file.write(document_preprocessing(train_data.data[i],train_data.target_names[train_data.target[i]])+'\n')

with open('preprocess_data/test_data.txt','w') as train_data_file:
    for i in range(len(test_data.data)):
      train_data_file.write(document_preprocessing(test_data.data[i],test_data.target_names[test_data.target[i]])+'\n')