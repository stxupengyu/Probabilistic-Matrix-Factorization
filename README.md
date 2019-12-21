# Probabilistic-Matrix-Factorization
Probabilistic Matrix Factorization for Recommendation by R    
我使用R语言实现了矩阵分解、概率矩阵分解算法。  
优化部分采用了随机梯度下降算法以及动量优化算法。  
本代码的核心部分改编自Ruslan Salakhutdinov提供的matlab代码（http://www.utstat.toronto.edu/~rsalakhu/BPMF.html）。  
除了输出训练集测试集误差，我还选取了部分预测评分与真实评分进行比较，显示了非常好的预测性能！  
除了预测评分，我还考虑了TOP-N推荐，最后可以为每位用户i推荐他最可能感兴趣的j部电影。  
最后，我封装了PMF函数（fun_pmf.R），通过多次调用该函数，我对比了概率矩阵分解算法的超参数对算法效果的影响(comparison.Rmd)，包括用户、物品隐特征矩阵维度k，学习率epsilon，正则化参数lambda，动量优化参数momentum。  
我使用的是MovieLen100k数据集(https://grouplens.org/datasets/movielens/) 包括1682名用户对943部电影的评分信息，共有100,000条评分数据。  
大部分注释是作者在学习Ruslan Salakhutdinov的代码时加上的，且为中文注释，英文阅读者可以参考Ruslan Salakhutdinov提供的matlab代码。  
为了方便展示，除了提供.R文件，我还提供了.Rmd文件。  
****
![Image text](https://github.com/stxupengyu/Probabilistic-Matrix-Factorization/blob/master/img-folder/comparison.png)
****
I use R to achieve matrix factorization and probability matrix factorization algorithm.  
In the optimization part, gradient descent algorithm and momentum optimization algorithm are used.  
The core part of this code is adapted from matlab code provided by Ruslan salakhutdinov (http://www.utstat.toronto.edu/~rsalakhu/BPMF.html).  
In addition to the output training set test set error, I also selected some prediction scores to compare with the real scores, showing a very good prediction performance!  
In addition to the prediction score, I also consider the top-N recommendation. Finally, I can recommend the most likely j movies for each user i.  
Finally, I encapsulate the PMF function（fun_pmf.R）. By calling this function many times, we compare the influence of the super parameters of the probability matrix factorization algorithm on the algorithm effect(comparison.Rmd), including the dimension k of the hidden feature matrix of users and items, the learning rate epsilon, the regularization parameter lambda, and the momentum optimization parameter momentum.  
I use the movielen100k data set(https://grouplens.org/datasets/movielens/), including the rating information of 1682 users for 943 movies, with a total of 100,000 rating data.  
Most of the comments are added by the author when learning the code of Ruslan salakhutdinov, and are in Chinese. For English readers, please refer to the matlab code provided by Ruslan salakhutdinov.  
For the convenience of presentation, in addition to the .R file, I also provide the .Rmd file.  
****
Reference:  
[1] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems.Computer 42, 8 (2009), 30–37.  
[2] R. Salakhutdinov and A. Mnih. Probabilistic matrixfactorization. In Advances in Neural Information Processing Systems (NIPS), volume 20, 2007.  
[3] Ruder S. An overview of gradient descent optimization algorithms[J]. arXiv preprint arXiv:1609.04747, 2016.  
[4] R. Salakhutdinov and A. Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo.In Proceedings of the Twenty-Fifth International Conference on Machine Learning (ICML 2008), Helsinki,Finland, 2008.  
