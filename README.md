# Comparative-Analysis-of-Collaborative-Filtering-Algorithms-for-Recommendation-Systems

## Get Started
This project leverages 100,836 ratings from 610 users across 9,742 movies to recommend films using five algorithms: 
- Matrix Factorization (MF)
- Tuned Matrix Factorization
- Funk SVD with RPCA
- Neural Collaborative Filtering (NCF)
- NCF with Contrastive Learning

## What You'll Achieve
- **Clean Data**: Filter sparse ratings and split data per user (**70% train, 15% CV, 15% test**).
- **Optimize Models**: Tune hyperparameters to combat overfitting and sparsity.
- **Interpret Results**: Evaluate model performance with ranking metrics like **Recall@20 and NDCG@20**.
- **Compare Performance**: Analyze accuracy, precision@20, recall@20, F1-score@20, and NDCG@20.

### 2. Download the Dataset
The notebook automatically downloads the **MovieLens ml-latest-small** dataset from GroupLens during execution. Alternatively, manually download it from [GroupLens](https://grouplens.org/datasets/movielens/) and extract it to `data/ml-latest-small/`.

## Understand the Results
The models were trained on a filtered dataset (~90,274 ratings) with a **70% train, 15% cross-validation, and 15% test split**. Check out the test set performance:

| Model                          | Accuracy (±0.5/±0.25) | Precision@20 | Recall@20 | F1-Score@20 | NDCG@20 |
|--------------------------------|----------------------|--------------|-----------|-------------|----------|
| **Matrix Factorization**       | 0.4088 (±0.5)       | 0.0716       | 0.3712    | 0.2400      | 0.2311   |
| **Tuned Matrix Factorization** | 0.4103 (±0.5)       | 0.0719       | 0.3739    | 0.2377      | 0.2372   |
| **Funk SVD + RPCA**            | 0.2251 (±0.25)      | 0.0621       | 0.2854    | 0.2667      | 0.1804   |
| **NCF**                        | 0.2476 (±0.25)      | 0.0757       | 0.3658    | 0.2658      | 0.2390   |
| **NCF + Contrastive Learning** | 0.1109 (±0.25)      | 0.0730       | 0.3947    | 0.2342      | 0.2478   |

## Take Action
- Choose **NCF + Contrastive Learning** for superior ranking (**Recall@20: 0.3947**, **NDCG@20: 0.2478**).
- Use **Tuned MF** for balanced performance with simpler computation (**Recall@20: 0.3739**).
- **Visualize Insights**: Run the notebook to see rating distributions and model comparisons.

## Compare with Research
See how this project stacks up against published studies:

| Paper                        | Model  | Dataset         | Metrics   | Why We're Different |
|------------------------------|--------|----------------|-----------|---------------------|
| **Xiangnan et al. (2017)**   | NeuMF  | MovieLens 1M   | HR@10: 0.71, NDCG@10: 0.426-0.450 | Our **NCF + Contrastive Learning** (**Recall@20: 0.3947**, **NDCG@20: 0.2478**) on smaller **ml-latest-small** dataset shows competitive ranking with simpler architecture. |
| **Koren et al. (2009)**      | SVD++  | Netflix (100M) | RMSE: 0.86 | Our **Funk SVD + RPCA** (**Recall@20: 0.2854**) underperforms due to smaller dataset but benefits from RPCA for sparsity. |

## Take It Further
Improve the system with these next steps:
- **Incorporate Content Features**: Add movie genres for hybrid recommendations.
- **Advanced Contrastive Learning**: Experiment with **SimCLR** for better embeddings.
- **Hyperparameter Optimization**: Use **grid search** or **Bayesian optimization** for tuning.
- **Attention Mechanisms**: Implement in NCF to capture complex interactions.
- **Ensemble Models**: Combine **MF, SVD, and NCF** for improved performance.
- **Scale Up**: Test on **larger datasets** like **MovieLens 1M** for generalizability.

## License
Licensed under the **MIT License**. See `LICENSE` for details.

## 📚 References & Resources
- **Dataset**: Harper, F.M., & Konstan, J.A. (2015). *"The MovieLens Datasets: History and Context."* ACM Transactions on Interactive Intelligent Systems. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)
- **Xiangnan et al. (2017)** *"Neural Collaborative Filtering."* DOI: [10.48550/arXiv.1708.05031](https://arxiv.org/abs/1708.05031)
- **Koren et al. (2009)** *"Matrix Factorization Techniques for Recommender Systems."* IEEE Computer. DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
- **Libraries**:
  - McKinney, W. (2010). *"Data Structures for Statistical Computing in Python."* Proceedings of the 9th Python in Science Conference.
  - Harris, C.R., et al. (2020). *"Array Programming with NumPy."* Nature, 585, 357–362.
  - Pedregosa, F., et al. (2011). *"Scikit-learn: Machine Learning in Python."* Journal of Machine Learning Research, 12, 2825-2830.
  - Paszke, A., et al. (2019). *"PyTorch: An Imperative Style, High-Performance Deep Learning Library."* Advances in Neural Information Processing Systems, 32.
  - Abadi, M., et al. (2016). *"TensorFlow: A System for Large-Scale Machine Learning."* 12th USENIX Symposium on Operating Systems Design and Implementation, 265-283.
  - Hunter, J.D. (2007). *"Matplotlib: A 2D Graphics Environment."* Computing in Science & Engineering, 9(3), 90-95.
  - Waskom, M. (2021). *"Seaborn: Statistical Data Visualization."* Journal of Open Source Software, 6(60), 3021.


## Need Help?
Reach out:

📧 Email: **[ghanbarpourshiadeh.amirali@gmail.com]**
