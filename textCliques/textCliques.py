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

from textCliques.textCleaning import remove_noise
from textCliques.dataIO import detect_duplicates, remerge_cliqueinfo
from textCliques.laserEmbed import calculate_multiling_laser
from textCluques.cosSim import get_pairwise_cossim, filter_cosSim_matrix, get_nonzero_similarity_tuples
from textCluques.network import create_edges, create_network, calculate_cliques

from pandarallel import pandarallel
pandarallel.initialize()


class textCliques:
    
    def __init__(self, cossim_threshold=0.9, min_ndegree=5, max_ndegree=500):
        
        self.cossim_threshold = cossim_threshold
        self.min_ndegree = min_ndegree
        self.max_ndegree = max_ndegree
        
        
    def cliqueFinder(self, 
                     full_data=None, 
                     languageColumn='lang', 
                     clean_text=True, 
                     dedupe_text=True,
                     idColumn='postUrl',
                     uncleaned_text_column_name='fulltext',
                     cleaned_text_column_name='clean_text',
                     multiprocess=True,
                     min_charlength=50,
                     min_cleantext_charlength=50,
                     max_cleantext_charlength=600,
                     min_ndegree=5,
                     max_ndegree=500
                    ):
        
        all_columns = list(full_data)
        
        if len(list(full_data)) < 2:
            raise Exception("Need a valid dataframe with at least an id and text column")
            #print("Need a valid dataframe with at least an id and text column")
            
        
            
        if languageColumn not in all_columns:
            raise Exception("Dataframe needs a column with language code for each row. Can be specified with languageColumn parameter in the cliqueFinder function call")
            
            
        if clean_text==True:
            print("cleaning text")
            
            if multiprocess==True:
                full_data['clean_text'] = full_data[uncleaned_text_column_name].parallel_apply(lambda x: remove_noise(x) if len(x) > min_charlength else '0')
                
                #remove lines that dont make the character length cut
                full_data = full_data[full_data['clean_text']!='0']
                
                #calculate length of cleaned text
                full_data['clean_textlen'] = full_data['clean_text'].parallel_apply(lambda x: len(x))
                
                full_data = full_data[(full_data['len_cleantext']>min_cleantext_charlength)&(full_data['len_cleantext']<max_cleantext_charlength)]
                
            else:
                full_data['clean_text'] = full_data[uncleaned_text_column_name].apply(lambda x: remove_noise(x) if len(x) > min_charlength else '0')
                
                #remove lines that dont make the character length cut
                full_data = full_data[full_data['clean_text']!='0']
                
                #calculate length of cleaned text
                full_data['len_cleantext'] = full_data['clean_text'].apply(lambda x: len(x))
                
                full_data = full_data[(full_data['len_cleantext']>min_cleantext_charlength)&(full_data['len_cleantext']<max_cleantext_charlength)]
                
                
            cleaned_text_column_name = 'clean_text'
            
            
        else:
            print("using {} as column for cleaned text".format(cleaned_text_column_name))
            
            #calculate length of cleaned text
            full_data['len_cleantext'] = full_data[cleaned_text_column_name].apply(lambda x: len(x))
            
            full_data = full_data[(full_data['len_cleantext']>min_cleantext_charlength)&(full_data['len_cleantext']<max_cleantext_charlength)]
            
            
        print("deduplicating the data")
        
        text2id, reduced_data, full_data, textCorp = detect_duplicates(full_data, idColumn=idColumn, cleanTextColumn=cleaned_text_column_name)
        
        
        print("calculating LASER multilingual embeddings")
        
        langCodes = reduced_data[languageColumn].tolist()
        
        embeddings = calculate_multiling_laser(textCorp, lang=langCodes)
        
        
        print("getting pairwise cosine similarity")
        
        sample_cosSim = get_pairwise_cossim(embeddings)
        
        print('filtering pairwise cosine similarity matrix based on cossim_threshold')
        
        _cossim_threshold = self.cossim_threshold
        
        sample_cosSim = filter_cosSim_matrix(sample_cosSim, threshold_similarity=_cossim_threshold)
        
        tot_nonzero, nonzero_entries = get_nonzero_similarity_tuples(sample_cosSim)
        
        print('reticulating the textverse...')
        
        print('calculating edges')
        
        edges = create_edges(tot_nonzero,nonzero_entries)
        
        print('building networkx graph')
        
        G = create_network(edges, min_degree =self.min_ndegree, max_degree=self.max_ndegree)
        
        print('calculating cliques')
        
        clique2nodes, node2cliques = calculate_cliques(G)
        
        
        print('merging the clique information with original data')
        
        textgroups = remerge_cliqueinfo(node2cliques, reduced_data, text2id, full_Data, idColumn=idColumn, cleanTextColumn=cleaned_text_column_name)
        
        print('all done!')
        
        return textgroups
        
        
        
        
        
        
            
            
                
                
            
            
                
            
        
            
                
                
                
                
                