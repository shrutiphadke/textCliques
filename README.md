# textCliques
Finding large groups of similar texts across languages




## Installation

#### 1. Install libraries from requirements.txt
#### 2. Download this repository


## Use
#### Example use

full_data = #read full data in pandas frame

##### initiate textClique instance with required threashold for cosine similarity, minimum degree and maximum degree for texts to be included in the network
tClique = textCliques(cossim_threshold=0.9, min_ndegree=5, max_ndegree=50)


##### use cliqueFinder function on the dataset, specify language for each data point in a separate column in the dataset
textgroups = tClique.cliqueFinder(full_data=full_data, languageColumn='language')

##### save data enumerated with clique numbers
textgroups.to_csv("outout.csv")

