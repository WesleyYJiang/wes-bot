import os
import json
import codecs
import csv

directory = "/Users/wesleyjiang/Downloads/self_dialogue_corpus-master/dialogues"
count = 0
lines = []
for root,dirs,files in os.walk(directory):
    for name in files:
        with open(os.path.join(root, name)) as f:
            for line in f:
                count = count + 1
                lines.append(line)
print(count)
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

datafile = os.path.join("self_dial_formatted.txt")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for i in range(372720):
        if i%2 == 0:
            first = lines[i]
            first = first.replace('"', '')
            first = first.rstrip()
            second = lines[i+1]
            second = second.replace('"', '')
            second = second.rstrip()
            writer.writerow([first, second])