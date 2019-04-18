import csv
import os
import codecs
from io import open

count = 0
with open(os.path.join("/Users/wesleyjiang/Downloads/dial3.txt"), 'r') as datafile:
    lines = []
    for line in datafile:
        line = line.rstrip()
        lines.append(line)
        count = count+1
print(count)

# delimiter = '\t'
# delimiter = str(codecs.decode(delimiter, "unicode_escape"))
#
# datafile = os.path.join("dia-formatted.txt")
#
# with open(datafile, 'w', encoding='utf-8') as outputfile:
#     writer = csv.writer(outputfile, delimiter=delimiter)
#     for i in range(152104):
#         if i%2 == 0:
#             writer.writerow([lines[i], lines[i+1]])