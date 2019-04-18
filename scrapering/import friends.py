from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


with open(os.path.join("friends.txt"), 'r') as datafile:
    lines = []
    for line in datafile:
        line.replace('"', '')
        line = line.rstrip()
        lines.append(line)

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

datafile = os.path.join("friends-formatted.txt")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for i in range(60849):
        writer.writerow([lines[i], lines[i+1]])