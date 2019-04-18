import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


with open(os.path.join("the-office-all-episodes.txt"), 'r') as datafile:
    lines = []
    for line in datafile:
        line=line.replace('"', '')
        line=line.replace('-', '')
        line = line.rstrip()
        lines.append(line)

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

datafile = os.path.join("office-formatted.txt")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for i in range(76475): 
        writer.writerow([lines[i], lines[i+1]])