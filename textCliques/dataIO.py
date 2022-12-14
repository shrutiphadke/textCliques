import os
import time
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import glob
import json
from random import sample
import re
import string
import warnings


    
def detect_duplicates(textFrame, idColumn='postUrl', cleanTextColumn='clean_text'):
    """
    Args:
    textFrame :      cleaned text frame with at least these three columns: id, clean text.
    idColumn:        column to be used as an identifier for each text
    cleanTextColumn: column with cleaned text
    
    Returns:         text2id: dataframe mapping clean text to all its duplicate versions
                     reduced_data dataframe: duduped version of the original data
                     full_data: original dataframe
                     textCorp: list of texts to be used for creating embeddings
                     
    """
    
    text2id = textFrame.groupby(cleanTextColumn)[idColumn].apply(list).reset_index()
    full_data = textFrame.copy()
    reduced_data = textFrame.drop_duplicates(subset=[cleanTextColumn])
    reduced_data['tno'] = list(range(len(reduced_data)))
    
    textCorp = reduced_data[cleanTextColumn].tolist()
    
    return text2id, reduced_data, full_data, textCorp



def remerge_cliqueinfo(node2cliques, reduced_data, text2id, full_Data, idColumn='postUrl', cleanTextColumn='clean_text'):
    """
    Args:
    textFrame :      cleaned text frame with at least these three columns: id, clean text, unclean text.    
    reduced_data:    deduped form of textframe generated with detect_duplicates    
    text2id :        clean text to its duplicate versions mapped by ids    
    full_data :      original complete dataset    
    idColumn :       column to be used as an identifier for each text    
    cleanTextColumn: column with cleaned text
    
    Returns:         dataframe with column "clique" demarking the clique to which the row text belongs to

    """
    
    reduced_data['component'] = reduced_data['tno'].apply(lambda x: node2cliques[x] if x in node2cliques else 'nan')
    clique_data = final_data[final_data['component']!='nan']
    text2tno = clique_data.set_index(cleanTextColumn).to_dict()['component']
    
    url2component = defaultdict(int)
    texts_with_no_cliques = []
    for index, row in text2id.iterrows():
        k = row[cleanTextColumn]

        urlList = row[idColumn]
        for u in urlList:
            try:
                url2component[u] = text2tno[k]
            except:
                texts_with_no_cliques.append(k)
                
    nondup_url2component = clique_data.set_index(idColumn).to_dict()['component']
    url2component.update(nondup_url2component)
    full_data['clique'] = full_data[idColumn].apply(lambda x: url2component[x] if x in url2component else 'nan')
    textgroups = full_data[full_data['clique']!='nan']
    return textgroups

    
        
        
