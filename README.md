# Recommendation System
This project explores different methods of recommendation system and evaluate their performances on two different datasets.

# Datasets
## Movie Ratings
The dataset used for collaborative filtering is [movie-ratings](https://drive.google.com/drive/folders/1_JF9plSjE3PAFBuSvUFRkDdftJWo1TFz). 
The file `ratings.csv` contains the rows including a rating of a user on a movie. Its columns are 
- `userId` 
- `movieId`
- `rating` 
- `timestamp`

which are essential for constructing the sparse rating matrix $R$. 
Other files contain more detailed information for exploration. For example, `tags.csv` contains the movie tags given by users, and `movies.csv` contains the movie genres.

## MSLR-WEB10K
The data used for ranking system is [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) and helper code is provided for data processing. 
A dataset in LTR usually contain queries, a list of documents associated with each query, and relevance judgements/scores for each query-document pair, where

- relevance is often graded on a scale
- a query-document pair describes the relationship between a query and a document, often represented by a feature vector
  - e.g., TF-IDF scores, page ranks, user click-through rates, ......

For example, this row from MSLR-WEB30K 
```
0 qid:1 1:3 2:0 3:2 4:2 ... 135:0 136:0
```
has the following meaning:
- query: the one with id 1
- query-document pair: has a 136-d feature
- relevance score: 0

# Methods
## User-based collaborative filtering
### k-NN
The essential idea is to find similar users to the target user and recommend based on how they liked the target movie not rated by the target user. This is also taken as k-NN method.

We need Pearson correlation coefficient between user $u$ and $v$:
$$\text{Pearson}(u,v) = \frac{\sum_{k\in I_u\cap I_v}(r_{uk} - \mu_u)(r_{vk} - \mu_v)}{\sqrt{\sum_{k\in I_u\cap I_v}(r_{uk} - \mu_u)^2}\sqrt{\sum_{k\in I_u\cap I_v}(r_{vk} - \mu_v)^2}}$$
- $I_u$: the set of item indices for which ratings have been specified by user $u$
- $I_v$: the set of item indices for which ratings have been specified by user $v$
- $\mu_u$: mean rating for user $u$ computed using its specified ratings
- $r_{uk}$: rating for user $u$ for item $k$

Then, the predicted rating of user $u$ for item $j$ is
$$\hat{r}\_{uj} = \mu_u + \frac{\sum_{v\in P_u}\text{Pearson}(u,v)(r_{vj} - \mu_v)}{\sum_{v\in P_u}|\text{Pearson}(u,v)|}$$
where $P_u$ is the neighbourhood of the target user $u$.
This prediction simulates the variance of ratings from those users in the target neighbourhood based on the target user’s mean rating.

Notice that the computation of Pearson coefficients is intensive as the offline phase, while the prediction is easy as the online phase.

### Naive mean rating 
This is an experimental method, which simply takes the mean rating of a user as the predicted rating on a target movie by this user.

## Model-based collaborative filtering
The essence is to exploit the correlation between the rows and columns of rating matrix $R$. $R$ has built-in redundancies (common features of items), making it possible to approximate it well by product of low-rank factors.

The problem can be formulated as 
$$\underset{U,V}{\min} \hspace{1ex} \frac{1}{2}\sum_{(i,j)\in S} (r_{ij} - \sum_{s=1}^k u_{is} v_{js})^2$$
where $S=\{(i,j): r_{ij} \text{ is observed}\}$.

I explored two variants of the matrix factorization methods.
### Non-negative Matrix Factorization (NMF)
This enforces the constraint $U\geq0, V\geq0$. 
The prediction of user $i$ on movie $j$ is 
$$\hat{r}\_{ij} = \sum_{s=1}^k u_{is} v_{js}$$

The advantage is that it provides high interpretability for user-item interactions. With the NMF, we can measure the features of a new item based on its content, brand, … to evaluate its distribution in the latent feature space. Then, we know the predicted ratings from all the users on this new item.

### Matrix Factorization with bias
This method adds the L2 regularization and bias for each user and movie. The formulated problem is $$\underset{U,V, b_u, b_i}{\min} \hspace{1ex} \frac{1}{2}\sum_{(i,j)\in S} (r_{ij} - \sum_{s=1}^k u_{is} v_{js})^2 + \frac{\lambda}{2}||U||\_F + \frac{\lambda}{2}||V||\_F + \frac{\lambda}{2}\sum_{u=1}^m b_u^2 + \frac{\lambda}{2}\sum_{i=1}^n b_i^2$$
where $b_u$ is the bias of user $u$ and $b_i$ is the bias of item $i$.

The prediction of user $i$ on movie $j$ is 
$$\hat{r}\_{ij} = \sum_{s=1}^k u_{is} v_{js} + \mu + b_i + b_j$$

## Learning-To-Rank
In this project, I used the pairwise approach to learn the ranking, where the ranking problem is viewed as a problem of correctly ordering pairs of items and aim to minimize the number of incorrectly ordered pairs.

The model is implemented in `LightGBM` with  objective=`lambdarank`. 

## Pairwise Approach
Assume for a given query, we have items $x_i, x_j$ with their true relevance scores labeled as $y_i, y_j$ respectively. We find  a scoring function $f(x)$ that predicts the relevance scores $s_i, s_j$ for $x_i, x_j$.

Now, we have a pair of items $(x_i, x_j)$ with scores $(s_i, s_j)$. To calculate the error based on the order of this pair of items, define the loss function $Loss(s_i, s_j)$ by approximating this problem to a binary classification task.

The true probability $P_{ij}$ is defined
```math
P_{ij} = \begin{cases}
1 & \text{if } y_i > y_j\\
0 & \text{else}
\end{cases}
```

The predicted probability $\bar{P}\_{ij}$ is computed $$\bar{P}_{ij} = \frac{e^{s_i - s_j}}{1 + e^{s_i - s_j}}$$

The loss is defined as the cross-entropy loss between $P_{ij}$ and $\bar{P_{ij}}$
$$Loss(s_i, s_j) = -[P_{ij}\log \bar{P_{ij}} + (1-P_{ij})\log(1-P_{ij})]$$

# Metrics
In the movie ratings dataset, the average RMSE and MAE are used to evaluate the performance.

In the WEB10K dataset, the normalized Discounted Cumulative Gain (nDCG) is used to evaluate the performance.

# Execution of code
The notebook can be directly run on Colab. GPU is recommended. Notice that the LTR part requires high RAM resources.

The data folders `movie_data` and `MSLR-WEB10K` should be placed on the same level as the notebook file.
