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
import time
import datetime
import codecs
import logging
import os
import sys
import argparse
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
    
    
#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def percentage(v):
    try:
        v = float(v)
        
        if v >= 0 and v <= 1:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")    
    
################################################################################

#Run:
#python3 bag2bag.py --input output/bov/txt/ --output output/bov/txt/

#Defining script arguments: 
parser = argparse.ArgumentParser(description="Convert a Doc-Term 'cat-pol' to Doc-Term 'cat' and 'pol'.")
parser.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process - def. False')
parser.add_argument("--split", metavar='PATH', type=str, action="store", dest="split", required=True, nargs="?", const=True, help='special "token" to split classes')
parser.add_argument("--input", "-i", metavar='PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of Doc-Term files')
parser.add_argument("--output", "-o", metavar='PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save Doc-Term files')
args = parser.parse_args()    #Verifying arguments.

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("")
total_start = time.time()

################################################################################
### INPUT (LOADING FILES LIST)                                               ###
################################################################################

if not os.path.exists(args.input):
    print("ERROR: input directory does not exists!")
    print("\t!Directory: " + args.input) 
    sys.exit()
     
print("> Loading input files...\n")
files = []
filesList = []
 
#Loading all files from root directory:     
for catpol_fileItem in os.listdir(args.input):
    filesList.append(args.input + catpol_fileItem)
              
filesList.sort()
filePath_i = 0
total_num_examples = len(filesList)
eta = 0
print("> Converting input files:")
print("..................................................")
print_progress(filePath_i, total_num_examples, eta)
             
#Reading files:
for filePath in filesList:
    start = time.time()
    catpol_fileItem = codecs.open(filePath, "r", "utf-8")
    cat_fileItem = codecs.open(filePath.replace("_cat-pol_", "_cat_"), "w", "utf-8")
    pol_fileItem = codecs.open(filePath.replace("_cat-pol_", "_pol_"), "w", "utf-8")
    firstLine = catpol_fileItem.readline()
    header = catpol_fileItem.readline()
    cat_fileItem.write(firstLine)
    pol_fileItem.write(firstLine)
    cat_fileItem.write(header)
    pol_fileItem.write(header)
    lines = catpol_fileItem.readlines()
    
    for line in lines:
        cells = line.split("\t")
        class_atr = cells[-1].replace("\n", "")
        newLine = '\t'.join(cells[:-1])
        cat_fileItem.write(newLine + "\t" + class_atr.split(args.split)[0] + "\n")
        pol_fileItem.write(newLine + "\t" + class_atr.split(args.split)[1] + "\n")
    
    catpol_fileItem.close()
    cat_fileItem.close()
    pol_fileItem.close()
    filePath_i += 1
    end = time.time()
    eta = (total_num_examples-filePath_i)*(end-start)
    print_progress(filePath_i, total_num_examples, eta)
     
print("..................................................\n")

################################################################################


################################################################################

total_end = time.time()
print("> Log:")
print("..................................................")
print("- Time: " + str(format_time(total_end-total_start)))
print("- Output files: " + str(total_num_examples*2))
print("..................................................\n")
