import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import glob
import json
from random import sample
import sklearn
import re
import string
import warnings
from bs4 import BeautifulSoup
import datetime
import os
import time
import sys

from textCliques.textCliques import textCliques

from pandarallel import pandarallel
pandarallel.initialize()

full_data = pd.read_csv("/data/shruti/ONR/small_data/kheri_complete_unclean_data.csv")

tClique = textCliques(cossim_threshold=0.95, min_ndegree=5, max_ndegree=50)

textgroups = tClique.cliqueFinder(full_data=full_data, languageColumn='language')

textgroups.to_csv("/data/shruti/ONR/small_data/kheri_pythonwrapper_outout.csv")

