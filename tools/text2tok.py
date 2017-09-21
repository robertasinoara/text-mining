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
#python3 text2tok.py --input input/dataset/ --output input/dataset/tokenized/

#Defining script arguments: 
parser = argparse.ArgumentParser(description="Convert raw texts to tokenized texts.")
parser.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process - def. False')
parser.add_argument("--input", "-i", metavar='PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory to load raw texts')
parser.add_argument("--output", "-o", metavar='PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save tokenized texts')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
print("")
total_start = time.time()
texts_dir = ""
ids_dir = ""
babelfy_requests = 0

################################################################################


################################################################################

log = codecs.open("text2tok-log_" + str( int(time.time()) ) + ".txt", "w", "utf-8")
print("> Removing empty lines:")
print("..................................................")
log.write("> Raw texts: " + args.input + "\n")
files_list = []

#Reading all files from all root directories:     
for directory in os.listdir(args.input):
    for file_item in os.listdir(args.input + "/" + directory):
        files_list.append(args.input + directory + "/" + file_item)
             
files_list.sort()
total_num_examples = len(files_list) 
filepath_i = 0
eta = 0
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()

#Reading database:
for filepath in files_list:
    start = time.time()
    log.write("\t" + filepath + "\n")
    file_item = codecs.open(filepath, "r", "utf-8")
    paragraphs = [s.strip() for s in file_item.read().splitlines()]    #Removing extra spaces
    file_item.close()
    file_item = codecs.open(filepath, "w", "utf-8")
    
    for paragraph in paragraphs:
        if not paragraph.strip(): continue    #Ignoring blank line.
        file_item.write(paragraph + "\n")

    file_item.close()
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)   
    
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n")
print("> Creating directory for tokenized texts...\n")
tokenized_texts_location = args.output + "tokenized_texts/" 
log.write("\n\n\n> Tokenized texts: " + tokenized_texts_location + "\n")
print("> Tokenizing raw texts:")
print("..................................................")
log.write("\tFiles: " + str(total_num_examples) + "\n\n")

for filepath in files_list:
    log.write("\t" + filepath + "\n")
     
filepath_i = 0
eta = 0
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()
log.write("\tFiles: " + str(total_num_examples) + "\n\n")

#Reading database:
for filepath in files_list:
    start = time.time()
    new_filepath = filepath.replace(args.input, tokenized_texts_location)
    log.write("\t" + new_filepath + "\n")
    file_item = codecs.open(filepath, "r", "utf-8")
    paragraphs = [s.strip() for s in file_item.read().splitlines()]    #Removing extra spaces
    file_item.close()
    
    for index, paragraph in enumerate(paragraphs):
        #The order is very important to extract knowledge:
        paragraphs[index] = nltk.tokenize.word_tokenize(paragraph)    #Work well for many European languages.          
        
    #Writing tokenized content to new file:
    new_dir = '/'.join( new_filepath.split("/")[:-1] ) + "/"
    
    if not os.path.exists(new_dir):
        os.makedirs(os.path.abspath(new_dir), mode=0o755)    #Creating intermediated directories
            
    new_file_item = codecs.open(new_filepath, "w", "utf-8")
                
    for paragraph in paragraphs:
        new_file_item.write(' '.join(paragraph) + "\n")

    new_file_item.close()
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)   
    
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n")
log.write("\n")
log.close()

################################################################################


################################################################################

total_end = time.time()
print("> Log:")
print("..................................................")
print("- Time: " + str(format_time(total_end-total_start)))
print("- Files: " + str(total_num_examples))
print("..................................................\n")
