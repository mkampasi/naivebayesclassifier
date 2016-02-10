import os
import string
import sys
import math
from nltk.stem import PorterStemmer
import codecs
stemmer=PorterStemmer()
mainfolder = "C:/USC Course material/CS - 544 NLP/Assignments/Assignment2/op_spam_train - new"
stopwordfilename = '/stopwords_en.txt'

#Initialize variables for priors and posterior probabilities
priorPD = 0.0
priorPT = 0.0
priorND = 0.0
priorNT = 0.0
vocabulary = []
postprobability_PD =[]
postprobability_PT = []
postprobability_ND = []
postprobability_NT = []

def checkEqual(lst):
       return lst[1:] == lst[:-1]
      
def remove_duplicates(seq):
#Remove duplicates from any list
#In: a list with duplicates
#Out: A list with duplicates removed
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

    
def parsefile(foldername):
#Function: parse all sub-folders in the folder.
#In: Top-level directory
#Out: A tuple - A string of text from all the parsed files concatenated and the number of files parsed 
    numberoffiles = 0
    dirlist=os.listdir(foldername)
    text = ""
    for x in dirlist:
        for directory, dirnames, files in os.walk(os.path.join(foldername,x)):
            for filename in files:
                with open(os.path.join(directory,filename)) as f:
                    text += ''.join(f.read())
                    text+=" "
                numberoffiles+=1
    return text,numberoffiles

def remove_stopwords(text):
#Function: Remove stop words from a list
#In: A list of words found in a class
#Out: A list of words with the stop words removed. Also, all words lower cased
    stopwordfilepath = os.getcwd() + stopwordfilename 
    with open(stopwordfilepath) as f:
        stopwords = [line.strip() for line in f]
    thistxttokens = [w.lower() for w in text if w.lower() not in stopwords and len(w) > 1]
    return thistxttokens


def frequency_selection(classet):
#Function: Eliminate words from the feature set of a class if the frequency of occurance is lesser than 5
#In: a list of features (words) for a class
#Out: A filtered list of features with the uncommon words removed
    classet_deduplicate = remove_duplicates(classet) 
    termcounts = [classet.count(word)  for word in classet_deduplicate] 
    freq_dict = dict(zip(classet_deduplicate, termcounts))
    freq_list_sorted = sorted(freq_dict.items(), key=lambda x: x[1],reverse=True)
    #freq_list_sorted =  freq_list_sorted[10:]
    freq_list_filtered = [(k,v) for k, v in freq_list_sorted if v>=5]
    #freq_list= list(freq_dict_filtered)
    #(key,value) =  zip(*freq_list)
    #classet_new =  [(word) for word in classet if word in freq_list_filtered] 
    #return classet_new
    return freq_list_filtered 

def calculate_postprobablity(vocab,classet):  
#Function: Calculate Posterior probabilities
#In: Entire Vocabulary list , a list of features (words) for a class
#Out: A list of posterior probability values calculated for each word in the vocabulary with respect to a class
    Tct = [(float)(classet.count(word) + 1) for word in vocab]
    #print dict(zip(vocab,Tct)) 
    Tctdash = sum(Tct)
    #Add a default value for an test word which is not present in the vocabulary
    TbyC = [math.log((tct/Tctdash)) for tct in Tct] 
    return TbyC    
        
def nblearn(mainfolder):
    #parse files for all four classes
    (posdeceptivetext,posdecep) = parsefile(mainfolder + "/positive_polarity/deceptive_from_MTurk")
    (postruthtext,postruth) = parsefile(mainfolder + "/positive_polarity/truthful_from_TripAdvisor")
    (negdeceptivetext,negdecep) = parsefile(mainfolder + "/negative_polarity/deceptive_from_MTurk")
    (negtruthtext,negtruth) = parsefile(mainfolder + "/negative_polarity/truthful_from_Web")
    totalfiles = posdecep + postruth + negdecep + negtruth
    
    #Calculate each prior probability
    priorPD = math.log(float(posdecep)/totalfiles) if posdecep!=0 else 0.0
    priorPT = math.log(float(postruth)/totalfiles) if postruth!=0 else 0.0
    priorND = math.log(float(negdecep)/totalfiles) if negdecep!=0 else 0.0
    priorNT = math.log(float(negtruth)/totalfiles) if negtruth!=0 else 0.0
 
    #Remove numbers
#     for i in range(0,4):
#         posdeceptivetext = ''.join([s for s in posdeceptivetext if not s.isdigit()])
#         postruthtext = ''.join([s for s in posdeceptivetext if not s.isdigit()])
#         negdeceptivetext = ''.join([s for s in posdeceptivetext if not s.isdigit()])
#         negtruthtext = ''.join([s for s in posdeceptivetext if not s.isdigit()])
#  
    #Convert string to list   
    PDtext =  posdeceptivetext.split()
    PTtext = postruthtext.split()
    NDtext = negdeceptivetext.split()
    NTtext = negtruthtext.split()
   
    #Remove stop words
    PDtext =  remove_stopwords(PDtext)
    PTtext =  remove_stopwords(PTtext)
    NDtext =  remove_stopwords(NDtext)
    NTtext =  remove_stopwords(NTtext)
    
    #Remove punctuations
    PDtext = [ch.translate(None,string.punctuation) for ch in PDtext]
    PDtext = filter(None, PDtext) 
    PTtext = [ch.translate(None,string.punctuation) for ch in PTtext]
    PTtext = filter(None, PTtext)
    NDtext = [ch.translate(None,string.punctuation) for ch in NDtext]
    NDtext = filter(None, NDtext)
    NTtext = [ch.translate(None,string.punctuation) for ch in NTtext]
    NTtext = filter(None, NTtext)
    
    
    #Remove uncommon words
    PDList = frequency_selection(PDtext)
    PTList = frequency_selection(PTtext)
    NDList= frequency_selection(NDtext)
    NTList = frequency_selection(NTtext)
    #TODO
    
    PDtext1 = [k for (k,v) in PDList]
    PTtext1 = [k for (k,v) in PTList]
    NDtext1 = [k for (k,v) in NDList]
    NTtext1 = [k for (k,v) in NTList]
    
    common_vocabulary = [commonword for commonword in PDtext1 if commonword in NDtext1 and commonword in PTtext1 and commonword in NTtext1]
    common_vocab = []
    
    for word in common_vocabulary:
        temp = []
        value1 = [v for (k,v) in PDList if k==word]
        value2 = [v for (k,v) in PTList if k==word]
        value3 = [v for (k,v) in NDList if k==word]
        value4 = [v for (k,v) in NTList if k==word]  
        #temp.append(value1+value2+value3+value4)
        common_vocab.append((word,value1+value2+value3+value4))
    
    print common_vocabulary
    print common_vocab
    common_vocab = []
    for word in common_vocabulary:
        value1 = [math.floor(float(v)/float(posdecep)) for (k,v) in PDList if k==word]
        value2 = [math.floor(float(v)/float(postruth)) for (k,v) in PTList if k==word]
        value3 = [math.floor(float(v)/float(negdecep)) for (k,v) in NDList if k==word]
        value4 = [math.floor(float(v)/float(negtruth)) for (k,v) in NTList if k==word]  
        common_vocab.append((word,value1+value2+value3+value4))
    
    print common_vocab
    words_to_remove=[]
    for key,list in common_vocab:
        if checkEqual(list):
            words_to_remove.append(key)
        
    PDtext2 =  [(word) for word in PDtext if word in PDtext1 and word not in words_to_remove] 
    PTtext2 =  [(word) for word in PTtext if word in PTtext1 and word not in words_to_remove] 
    NDtext2 =  [(word) for word in NDtext if word in NDtext1 and word not in words_to_remove] 
    NTtext2 =  [(word) for word in NTtext if word in NTtext1 and word not in words_to_remove] 
    
       
    #Generate Vocabulary    
    vocabulary = PDtext2 + PTtext2 + NDtext2 + NTtext2 
    vocabulary = remove_duplicates(vocabulary)
    
    #Calculate Post probabilities
    postprobability_PD = calculate_postprobablity(vocabulary,PDtext2)
    postprobability_PT = calculate_postprobablity(vocabulary,PTtext2)
    postprobability_ND = calculate_postprobablity(vocabulary,NDtext2)
    postprobability_NT = calculate_postprobablity(vocabulary,NTtext2)

    #Open file to output the NB model
    mymodel_file = open("mymodel.txt", "wb")
    
    #Convert to a dictionary of word,posterior prob values for each class
    dictionaryPD = dict(zip(vocabulary, postprobability_PD))
    dictionaryPT = dict(zip(vocabulary, postprobability_PT))
    dictionaryND = dict(zip(vocabulary, postprobability_ND))
    dictionaryNT = dict(zip(vocabulary, postprobability_NT))
    
    classes = ["PD","PT","ND","NT"] 
    priors = [priorPD,priorPT,priorND,priorNT]
    dictionary_priors = dict(zip(classes, priors))
    mymodel_file.write(str(dictionary_priors) + "\n")
    mymodel_file.write(str(dictionaryPD) + "\n")
    mymodel_file.write(str(dictionaryPT) + "\n")
    mymodel_file.write(str(dictionaryND) + "\n")
    mymodel_file.write(str(dictionaryNT) + "\n")


#if __name__ == "__main__":
 #   mainfolder = sys.argv[1]
nblearn(mainfolder)
  