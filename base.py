from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from os import path
import tempfile
import numpy as np
from six.moves import urllib

from tensorflow.python.platform import gfile

DataSet = collections.namedtuple('DataSet', ['data', 'target'])
DataSets = collections.namedtuple('DataSets', ['train', 'test'])