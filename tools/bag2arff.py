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


################################################################################

#Run:
#python3 bag2arff.py --weka weka.jar --input output/bov/txt/ --output output/bov/arff/

#Defining script arguments: 
parser = argparse.ArgumentParser(description="Convert a Doc-Term matrix to ARFF (Weka file).")
parser.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process - def. False')
parser.add_argument("--weka", metavar='PATH', type=str, action="store", dest="weka", required=True, nargs="?", const=True, help='file path to Weka jar API')
parser.add_argument("--input", "-i", metavar='PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of Doc-Term files')
parser.add_argument("--output", "-o", metavar='PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the ARFF files')
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
     
print("> Loading input files:")
print("..................................................")
files_list = []
 
#Loading all files from root directory:     
for file_item in os.listdir(args.input):
    files_list.append(args.input + file_item)
              
files_list.sort()
filepath_i = 0
total_num_examples = len(files_list)
eta = 0
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()
             
#Reading files:
for filepath in files_list:
    start = time.time()
    file_item = codecs.open(filepath, "r", "utf-8")
    tmp_file_item = codecs.open(filepath + ".tmp", "w", "utf-8")
    n = int( file_item.readline().split(" ")[1] )
    header = ""
        
    for d_i in range(1, n+1):
        header += "d" + str(d_i) + "\t"
       
    header += "class_atr\n"
    tmp_file_item.write(header)
    tmp_file_item.writelines(file_item.readlines())
    file_item.close()
    tmp_file_item.close()
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)
     
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True) 
print("..................................................\n")

################################################################################
 
 
################################################################################
### CONVERTING DOC-TERMs TO ARFFs                                                 ###
################################################################################
 
if not os.path.exists(args.output):
    print("> Creating directory to ARFFs...\n")
    os.makedirs(os.path.abspath(args.output), mode=0o755)
else:
    print("WARNING: Directory to save ARFFs already exists!\n")
     
print("> Converting Doc-Terms to ARFFs:")
print("..................................................")
filepath_i = 0
eta = 0
print_progress(filepath_i, total_num_examples, eta) 
operation_start = time.time()
  
#Reading files:
for filepath in files_list:
    start = time.time()
    file_name = filepath.split("/")[-1]
    arff_path = args.output + file_name + '.arff'
    os.system("java > /dev/null 2>&1 -classpath " + args.weka + " weka.core.converters.CSVLoader " + filepath  + ".tmp > " + arff_path + " -F '\t'")
    os.remove(filepath + ".tmp")
    arff_file = codecs.open(arff_path, "r", "utf-8")
    arff_name = arff_file.readline()[:-5]    #Removing ".tmp" extension.
    arff_content = arff_file.readlines()
    arff_file.close()
    arff_file = codecs.open(arff_path, "w", "utf-8")
    arff_file.write(arff_name + "\n")
    
    for line in arff_content:
        arff_file.write(line)
    
    arff_file.close()
    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)
    
operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n")
 
################################################################################


################################################################################

total_end = time.time()
print("> Log:")
print("..................................................")
print("- Time: " + str(format_time(total_end-total_start)))
print("- Files: " + str(total_num_examples))
print("..................................................\n")
