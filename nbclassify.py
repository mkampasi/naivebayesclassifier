import os
import string
import sys

#Read from mymodel.txt into a dictionary 
dicts_from_file = []
with open('mymodel.txt','r') as inf:
    for line in inf:
        dicts_from_file.append(eval(line))
            
mainfolder = "C:/USC Course material/CS - 544 NLP/Assignments/Assignment2/op_spam_train_test - new"
stopwordfilename = '/stopwords_en.txt'

def remove_stopwords(text):
    #Function: Remove stop words from the file data
    # In: a text (string)
    # Out: words of text in a list, lowercased and with stopwords removed
    stopwordfilepath = os.getcwd() + stopwordfilename 
    with open(stopwordfilepath) as f:
        stopwords = [line.strip() for line in f]
    thistxttokens = text.split()
    thistxttokens1 = [w.lower() for w in thistxttokens if w.lower() not in stopwords and len(w) > 1]
    return thistxttokens1

def checkmax(class1,class2,class3,class4):
    #Function: Find max of the likelihood probabilities for each class 
    # In: Float values for the likelihood of each class
    # Out: Float. Max of the 4 values
    
    maxnum = max(class1,class2,class3,class4)
    if maxnum == class1:
        return 'positive' , 'deceptive'
    elif maxnum == class2:
        return 'positive' , 'truthful'
    elif maxnum == class3:
        return 'negative' , 'deceptive'
    else:
        return 'negative' , 'truthful'

#parse all test data files
def parsefile(foldername):
    #Function: parse and classify all files in the sub-folders.
    #In: Top-level directory
    #Out: No return. Writes the file classification results to nboutput.txt 
    my_output_file = open("nboutput.txt", "wb")
    write_to_file = ""  
    text = ""
    #Loop through each file
    for dir, subdirs, files in os.walk(mainfolder):
       #Exclude files with extensions other than .txt and also README.txt
       files = [ fi for fi in files if (fi.endswith(".txt") and not (fi.startswith("README")) )   ]      
       for file in files:
        #Read prior values from mymodel.txt
        scorePD = dicts_from_file[0].get('PD')
        scorePT = dicts_from_file[0].get('PT')
        scoreND = dicts_from_file[0].get('ND')
        scoreNT = dicts_from_file[0].get('NT')
    
        with open(os.path.join(dir,file)) as f:
            text = ''.join(f.read())
            alltext = remove_stopwords(text)
            D = [ch.translate(None,string.punctuation) for ch in alltext]
            for word in D:
                try:                    
                    scorePD = scorePD + float(dicts_from_file[1].get(word))
                    scorePT = scorePT + float(dicts_from_file[2].get(word))
                    scoreND = scoreND + float(dicts_from_file[3].get(word))
                    scoreNT = scoreNT + float(dicts_from_file[4].get(word))
                except TypeError: #Words which are not found in the vocabulary are ignored
                    scorePD = scorePD +  0.0 
                    scorePT = scorePT +  0.0 
                    scoreND = scoreND +  0.0 
                    scoreNT = scoreNT +  0.0 
            
            (label1, label2) = checkmax(scorePD,scorePT,scoreND,scoreNT)
            #Write to file in correct format
            write_to_file = write_to_file +  label2 + " " + label1 + " " + dir + "/"  + file + "\n"
        my_output_file.write(write_to_file) #Write output to file  


      
#Parse the test data folder
if __name__ == "__main__":
  mainfolder = sys.argv[1]
  parsefile(mainfolder)
