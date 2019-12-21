# Probabilistic-Matrix-Factorization
Probabilistic Matrix Factorization by R  
我使用R语言实现了矩阵分解、概率矩阵分解算法。  
优化部分采用了随机梯度下降算法以及动量优化算法。  
本代码的核心部分改编自Ruslan Salakhutdinov提供的matlab代码（http://www.utstat.toronto.edu/~rsalakhu/BPMF.html）。  
除了输出训练集测试集误差，我们还选取了部分预测评分与真实评分进行比较，显示了非常好的预测性能！  
除了预测评分，我们还考虑了TOP-N推荐，最后可以为每位用户i推荐他最可能感兴趣的j部电影。  
我们使用的是MovieLen100k数据集(https://grouplens.org/datasets/movielens/) 包括1682名用户对943部电影的评分信息，共有100,000条评分数据。  
大部分注释是作者在学习Ruslan Salakhutdinov的代码时加上的，且为中文注释，英文阅读者可以参考Ruslan Salakhutdinov提供的matlab代码。  
为了方便展示，除了提供.R文件，我还提供了.Rmd文件。  
****
![Image text](https://github.com/stxupengyu/Probabilistic-Matrix-Factorization/blob/master/img-folder/comparison.png)
****
I use R to achieve matrix factorization and probability matrix factorization algorithm.  
In the optimization part, gradient descent algorithm and momentum optimization algorithm are used.  
The core part of this code is adapted from matlab code provided by Ruslan salakhutdinov (http://www.utstat.toronto.edu/~rsalakhu/BPMF.html).  
In addition to the output training set test set error, we also selected some prediction scores to compare with the real scores, showing a very good prediction performance!  
In addition to the prediction score, we also consider the top-N recommendation. Finally, we can recommend the most likely j movies for each user i.  
We use the movielen100k data set(https://grouplens.org/datasets/movielens/), including the rating information of 1682 users for 943 movies, with a total of 100,000 rating data.  
Most of the comments are added by the author when learning the code of Ruslan salakhutdinov, and are in Chinese. For English readers, please refer to the matlab code provided by Ruslan salakhutdinov.  
For the convenience of presentation, in addition to the .R file, I also provide the .Rmd file.  
****
Reference：
[1] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems.Computer 42, 8 (2009), 30–37.  
[2] R. Salakhutdinov and A. Mnih. Probabilistic matrixfactorization. In Advances in Neural Information Processing Systems (NIPS), volume 20, 2007.  
[3] Ruder S. An overview of gradient descent optimization algorithms[J]. arXiv preprint arXiv:1609.04747, 2016.  
[4] R. Salakhutdinov and A. Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo.In Proceedings of the Twenty-Fifth International Conference on Machine Learning (ICML 2008), Helsinki,Finland, 2008.  
