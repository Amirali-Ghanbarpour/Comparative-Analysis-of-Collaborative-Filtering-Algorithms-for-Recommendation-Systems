Movie Recommendation System

A movie recommendation system built using the MovieLens dataset (ml-latest-small) to predict user ratings and generate personalized recommendations. This project evaluates five algorithms: Matrix Factorization (MF), Tuned Matrix Factorization, Funk Singular Value Decomposition (SVD) with RPCA, Neural Collaborative Filtering (NCF), and NCF with Contrastive Learning. The focus is on optimizing Recall@20 and NDCG@20, with secondary metrics including accuracy, precision@20, and F1-score@20.

Table of Contents

Project Overview
Dataset
Requirements
Usage
Models and Implementation
Experimental Analysis
Results
Future Work
Contributing
License
Acknowledgments


Project Overview
This project tackles key challenges in building a recommendation system using the MovieLens dataset, which contains ~100,836 ratings from 610 users across 9,742 movies. The challenges addressed include:

Data Sparsity: ~1.7% matrix fill, mitigated by filtering users (≥20 ratings) and movies (≥5 ratings).
Cold-Start Problem: Focused on users and movies with sufficient ratings.
Overfitting: Used L2 regularization, weight decay, and early stopping.
Model Complexity: Balanced simple (MF, Funk SVD) and complex (NCF) models.
Generalization: Ensured via user-based data splitting (70% train, 15% CV, 15% test) and hyperparameter tuning.


Dataset
The MovieLens dataset (ml-latest-small) includes:

Ratings: 100,836 ratings (0.5–5.0 scale) from 610 users for 9,742 movies.
Movies: Movie IDs, titles, and genres.
Sparsity: ~1.7% of the user-movie matrix is filled.
Preprocessing: Filtered to ~90,274 ratings (users with ≥20 ratings, movies with ≥5 ratings).

Data Splitting:

Per-user split: 70% training, 15% cross-validation, 15% test.
Ensures user-specific patterns are preserved across splits.


Requirements
Install the required Python packages:
pip install numpy pandas torch tensorflow scikit-learn scipy matplotlib seaborn tqdm


Usage

Download the Dataset:

The notebook automatically downloads ml-latest-small.zip from GroupLens and extracts it to a data directory.
Alternatively, manually download and place it in the project directory.


Run the Notebook:

Open Recommendation_System_github_version.ipynb in a Jupyter environment (e.g., Google Colab).
The notebook includes:
Data loading and preprocessing.
Model training and evaluation for all algorithms.
Visualization of rating distribution (rating_distribution.png).




Key Files:

Recommendation_System_github_version.ipynb: Core implementation.
data/ml-latest-small/: Contains ratings.csv and movies.csv.
rating_distribution.png: Visualizes rating distribution.




Models and Implementation
The following algorithms were implemented:

Matrix Factorization (MF):

20-dimensional embeddings.
Optimized with Adam (lr=1e-3, weight decay=1e-4) over 200 epochs, minimizing MSE loss.


Tuned Matrix Factorization:

Enhanced MF with hyperparameter tuning for better performance.


Funk SVD with RPCA:

30-dimensional factors with biases, using RPCA for missing rating imputation.
Optimized via SGD (lr=0.02, reg=0.01) over 200 epochs.


Neural Collaborative Filtering (NCF):

Combines GMF and MLP branches with 64-dimensional embeddings and L2 regularization.
Trained with Adam (lr=0.001) over 50 epochs, using early stopping.


NCF with Contrastive Learning:

Extends NCF with InfoNCE contrastive loss for better embedding alignment.
Uses negative sampling, optimizing combined MSE and contrastive loss (alpha=0.01, lr=0.0005) over 50 epochs.




Experimental Analysis
To optimize the performance of Neural Collaborative Filtering (NCF) and NCF with Contrastive Learning, several hyperparameter experiments were conducted on the MovieLens ml-latest-small dataset, focusing on improving Recall@20 and NDCG@20.
NCF Experiments: The baseline NCF model achieved Recall@20 of 0.2906 and NDCG@20 of 0.1941 on the test set. The following parameter changes were tested:

Patience Parameter: Doubling the patience parameter increased training epochs from 8 to 20, improving Recall@20 to 0.3168 and NDCG@20 to 0.2155, indicating better convergence.
Dropout Parameter: Increasing dropout from 0.2 to 0.9 enhanced generalization, yielding Recall@20 of 0.3631 and NDCG@20 of 0.2297, a significant improvement over the baseline.
MLP Layers (3 to 5): Increasing MLP layers to 5 slightly improved performance to Recall@20 of 0.3640 and NDCG@20 of 0.2298, but added computational complexity.
MLP Layers (3 to 2): Reducing MLP layers to 2 provided the best results, with Recall@20 of 0.3651 and NDCG@20 of 0.2366, a 25.6% improvement in Recall@20 and 21.9% in NDCG@20 over the baseline, suggesting a shallower network better captures user-item interactions.
NeuMF Activation (None to Sigmoid): Adding sigmoid activation in NeuMF resulted in Recall@20 of 0.2944 and NDCG@20 of 0.2012, showing marginal improvement over the baseline but underperforming compared to other configurations.

NCF with Contrastive Learning Experiments: The baseline model achieved Recall@20 of 0.3494 and NDCG@20 of 0.2330. Optimized parameters improved performance to Recall@20 of 0.3748 and NDCG@20 of 0.2447, demonstrating the effectiveness of contrastive loss in enhancing embedding alignment.

Results



Model
Split
Accuracy (±0.5/±0.25)
Precision@20
Recall@20
F1-Score@20
NDCG@20



Matrix Factorization
Test
0.4088 (±0.5)
0.0716
0.3712
0.2400
0.2311


Tuned Matrix Factorization
Test
0.4103 (±0.5)
0.0719
0.3739
0.2377
0.2372


Funk SVD + RPCA
Test
0.2251 (±0.25)
0.0621
0.2854
0.2667
0.1804


NCF
Test
0.2476 (±0.25)
0.0757
0.3658
0.2658
0.2390


NCF + Contrastive Learning
Test
0.1109 (±0.25)
0.0730
0.3947
0.2342
0.2478



Best Model: NCF with Contrastive Learning excels in recommendation relevance (Recall@20=0.3947) and ranking quality (NDCG@20=0.2478).
Comparison with Literature:
NCF (He et al., 2017, DOI: 10.1145/3038912.3052569) on MovieLens 1M: HR@10=0.71, NDCG@10=0.426–0.450.
SVD++ (Koren et al., 2009, DOI: 10.1109/MC.2009.263) on Netflix: RMSE=0.86.
Our models are competitive on the smaller dataset, with contrastive learning boosting ranking performance.




Future Work

Integrate content-based features (e.g., movie genres) for hybrid recommendations.
Explore advanced contrastive learning techniques (e.g., SimCLR).
Optimize hyperparameters using grid search or Bayesian optimization.
Implement attention mechanisms in NCF for complex user-item interactions.
Test ensemble methods combining MF, SVD, and NCF.
Validate on larger datasets (e.g., MovieLens 1M).


Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch: git checkout -b feature/new-feature
Commit changes: git commit -m 'Add new feature'
Push to the branch: git push origin feature/new-feature
Open a pull request.


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

GroupLens for providing the MovieLens dataset.
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua (2017). Neural Collaborative Filtering. DOI: 10.1145/3038912.3052569.
Yehuda Koren, Robert Bell, and Chris Volinsky (2009). Matrix Factorization Techniques for Recommender Systems. DOI: 10.1109/MC.2009.263.

