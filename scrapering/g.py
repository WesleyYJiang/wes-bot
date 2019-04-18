import csv
import random
import re
import os
import unicodedata
import codecs
from io import open

corpus_name = "formatted_movie_lines"
corpus = os.path.join("scap", corpus_name)

with open(os.path.join("formatted_movie_lines.txt"), 'r') as datafile:
    lines = []
    for line in datafile:
        line.replace('"', '')
        lines.append(line)
        print(line)

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

datafile = os.path.join("formatted.txt")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for i in range(34450):
        if (i % 2) == 0:
            writer.writerow([lines[i], lines[i+1]])