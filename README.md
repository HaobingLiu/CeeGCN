# Title: Graph Structure Learning with Scale Contraction for Weighted Graph Clustering

Our paper has been accepted by WWW 2026.

# Requirements
dgl 1.0.1
h5py 2.10.0
igraph 0.10.4
keras 2.10.0
networkx 3.1.0
numpy 1.18.5
scipy 1.4.1
tensorflow 2.19.0
torch 2.3.0
scikit-learn 1.0.2
keras 2.15.0
python 3.8
cudatoolkit=10.1

# Code
entmax.py: Implementation of the activation function a-entmax.
FCM.py: Implementing fuzzy C-means clustering.
layers.py: Implementing a multi-headed attention mechanism.
losses.py: Implementing the loss function.
model.py: Multi-layer GAT learning node representation and clustering results.
metrics.py: Achievement of evaluation indicators ACC and F1.
trainer.py: Implementing the training of the model.
utils.py: Reading data to enable the construction of social networks.
run.py: Main function for training subgraphs.
re_run.py: Main function for training orginal graph.
sampling_center: Sampling cluster centers.
sampling_graph: Sampling subgraphs.
merge_subgraph: Merging the sampled subgraphs.

# Dataset(ML100k_CWG)
movie_gra.csv: <movie id1, movie id2, weight> movie id indicates the id of the movie and the weight represents the number of times two movies are consecutively clicked.
movie_info.csv: <movie id, label> label indicates the genre of the movie.

# raw-data(MovieLens 100K)
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are numbered consecutively from 1.  
              The data is randomly ordered. 
              This is a tab separated list of user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC  

u.item     -- Information about the items (movies); this is a tab separated list of movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once.
              The movie ids are the ones used in the u.data data set.

# Data Processing Procedure
1. From the u.data file, filter each user's rating records by user ID and sort these records by timestamp.
2. Compute weights based on each user's consecutive movie rating records. If a user has rated two adjacent movies, the weight between these two movies is increased by 1, as the user may be interested in similar movies.
3. Generate the movie_gra.csv file based on the above statistics.
4. Determine the unique genre for each movie from the u.item file. Count the frequency of all genres, and if a movie involves multiple genres, select the genre with the highest frequency as the movie's unique genre.
5. Filter all involved movie IDs from movie_gra.csv and use the unique genres obtained in step 4 as labels for these movies, generating the movie_info.csv file.

# Run
run.py to train the sampled graph and save the best model parameters. Then, input the original graph into the model to obtain the final clustering results.

