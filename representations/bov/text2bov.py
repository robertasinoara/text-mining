#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##      Developed (originally) by: João Antunes (joao4ntunes@gmail.com)       ##
##       Laboratory: labic.icmc.usp.br    Personal: joaoantunes.esy.es        ##
##                                                                            ##
##      "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga ##
################################################################################
 
from __future__ import print_function
import filecmp
import datetime
import argparse
import codecs
import logging
import nltk
import os
import sys
import time
import math


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='Progress:', decimals=1, bar_length=100, final=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        estimation  - Required  : iteration estimation in seconds (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    rows, columns = os.popen('stty size', 'r').read().split()
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)-len(prefix)-len(eta)-15
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s ETA %s' % (prefix, bar, percents, '%', eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')
        
    sys.stdout.flush()
    del rows


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")
    
    
#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)
        
        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")

################################################################################
        
        
################################################################################

#Run:
#python3 text2bov.py --n_gram 1 --model models/Google/GoogleVectors_300.txt --input input/dataset/tokenized/ --output output/bov/txt/

#Pre-trained word and phrase vectors (Google): https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#More info: https://code.google.com/archive/p/word2vec/

#Defining script arguments: 
parser = argparse.ArgumentParser(description="Create a Bag of Vectors based in a W2V model (text vectors).")
parser.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process - def. False')
parser.add_argument("--n_gram", metavar='NUM', type=natural, action="store", dest="n_gram", default=1, nargs="?", const=True, required=False, help='specify N-gram - def. 1')
parser.add_argument("--model", "-m", metavar='PATH', type=str, action="store", dest="model", required=True, nargs="?", const=True, help='input file_output of model (Word2Vec text vectors)')
parser.add_argument("--input", "-i", metavar='PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of files to test')
parser.add_argument("--output", "-o", metavar='PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the BoV')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("")
total_start = time.time()

################################################################################


################################################################################
### INPUT (LOADING FILES LIST)                                               ###
################################################################################

if not os.path.exists(args.input):
    print("ERROR: input directory does not exists!")
    print("\t!Directory: " + args.input) 
    sys.exit()

model = open(args.model, "r")
model.readline()
vectors = {}
model_dim = 0

#Loading model as an indexed dictionary:
for vector in model:
    data = vector.strip().split(' ')
    head = data[0].strip()
    data.pop(0)
    vectors[head] = [float(elt) for elt in data]
    model_dim = len(data)

################################################################################

print("> Loading input texts...\n")
files = []
files_list = []
 
#Loading all files from all root directories:     
for directory in os.listdir(args.input):
    for file_item in os.listdir(args.input + "/" + directory):
        files_list.append(args.input + directory + "/" + file_item)
              
files_list.sort()
total_num_examples = len(files_list)

################################################################################

out_string = args.output + "bov_cat-pol_ng"
header = str(total_num_examples) + " " + str(model_dim) + "\n"

for dim in range(1, model_dim+1):
    header += "d" + str(dim) + "\t"
    
header += "class_atr\n"
print("> TASK 1 - N-GRAM VARIATION / TASK 2 - TEXT REPRESENTATION:")
print("..................................................")
total_operations = args.n_gram*total_num_examples
filepath_i = 0
eta = 0
print_progress(filepath_i, total_operations, eta)
operation_start = time.time()

for n in range(1, args.n_gram+1):
    out_file = open(out_string + str(n), "w")
    out_file.write(header)    
    start = time.time()
    
    for file_item in files_list:
        words = []
        file_output = codecs.open(file_item, "r", "UTF-8")
        doc_vector = [0]*model_dim
        vectors_found = 0
        n_grams = list( nltk.everygrams(file_output.read().strip().split(" "), max_len=n) )
        
        for ng in n_grams:
            words.append("_".join(ng))
            
        file_output.close()
        class_atr = file_item.split('/')[-2].strip()
        
        #Sum all vectors found:
        for word in words:
            if word in vectors:
                doc_vector = [sum(x) for x in zip(*[doc_vector, vectors[word]])]
                vectors_found += 1
                
        #Dividing (arithmetic mean) final vector:        
        if vectors_found != 0:
            doc_vector = [x / vectors_found for x in doc_vector]
            
        out_file.write( "\t".join(str(e) for e in doc_vector) + "\t" + class_atr + "\n" )
        filepath_i += 1
        end = time.time()
        eta = (total_operations-filepath_i)*(end-start)
        print_progress(filepath_i, total_operations, eta)
            
    out_file.close()
    
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_operations, total_operations, eta, final=True)    
print("..................................................\n")
    
################################################################################    
    
#Comparing output files:
print("> Removing duplicated files:")
print("..................................................")

for i in reversed(range(2, args.n_gram+1)):     
    if (filecmp.cmp(out_string + str(i), out_string + str(i-1), shallow=False)):
        os.remove(out_string + str(i))
        print(out_string + str(i) + " \t\t\t--> REMOVED")
        
print("..................................................\n")
