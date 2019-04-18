import csv
import os
import codecs
from io import open


with open(os.path.join("howimetyourmother.txt"), 'r') as datafile:
    lines = []
    for line in datafile:
        line=line.replace('"', '')
        line=line.replace('- ', '')
        line = line.rstrip()
        lines.append(line)

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

datafile = os.path.join("himym-formatted.txt")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for i in range(69410):
        writer.writerow([lines[i], lines[i+1]])