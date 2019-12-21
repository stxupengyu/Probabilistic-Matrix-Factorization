fun_pmf <- function(epsilon,lambda,momentum,maxepoch,num_feat){
  #参数设置
  #rm(list=ls())
  epsilon=epsilon # Learning rate 学习率
  lambda  = lambda#Regularization parameter正则化参数 
  momentum=momentum#动量优化参数
  epoch=1#初始化epoch
  maxepoch=maxepoch#总训练次数
  err_train3=rep(0,maxepoch)
  err_valid3=rep(0,maxepoch)
  err_random=rep(0,maxepoch)
  
  
  #导入数据
  data=read.csv("ratings100k.csv", header = FALSE)
  
  
  rating=data[,3]
  movie=data[,2]
  user=data[,1]
  #hist(rating)
  data_num=length(rating)
  #打乱数据集
  #rr=sample(1:data_num,data_num)
  #data= data[rr,]
  
  
  train_num=(data_num*9)%/%10;
  train_vec=data[1:train_num,];#训练集
  probe_vec=data[train_num:data_num,];#测试集
  
  movie_num= max(movie)#电影数量1682
  user_num=max(user)#用户数量943
  #sum_list=max(data)
  mean_rating = mean(train_vec[,3])#平均评级
  
  pairs_tr = length((train_vec[,3]))#training data 训练集长度
  pairs_pr = length((probe_vec[,3]))#validation data 验证集长度
  
  numbatches= 9 #Number of batches  每次训练9个数据
  num_m = movie_num  # Number of movies 电影数量
  num_p = user_num  #Number of users 用户数量
  num_feat = num_feat #Rank 10 decomposition 隐因子数量
  #初始化
  w1_M1     = 0.1*matrix(runif(num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T) # Movie feature vectors 生成用户物品特征矩阵d=10
  w1_P1     = 0.1*matrix(runif(num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T) # User feature vecators
  w1_M1_inc = matrix(rep(0,num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T)#生成同shape的全零矩阵
  w1_P1_inc = matrix(rep(0,num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T)
  
  
  for(epoch in epoch:maxepoch){
    
    #rr=sample(1:pairs_tr,pairs_tr)
    #train_vec = train_vec[rr,];#将训练集的顺序打乱
    
    #采用mini batch的方法，每次训练9个样本
    for(batch in 1:numbatches){
      
      #print(c('epoch',epoch,'batch',batch))
      
      N=10000 #number training triplets per batch 每次训练三元组的数量
      
      aa_p= train_vec[((batch-1)*N+1):(batch*N),1]#读取用户列每次读取一万个
      aa_m= train_vec[((batch-1)*N+1):(batch*N),2]#读取电影列
      rating = train_vec[((batch-1)*N+1):(batch*N),3]#读取评级列
      rating = rating-mean_rating; #Default prediction is the mean rating. 
      
      # Compute Predictions %%%%%%%%%%%%%%%%%
      pred_out= apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum)#每一行进行求和,.*为对应元素相乘 10k*1，用隐特征矩阵相乘得到1w个预测评级
      f=sum((pred_out - rating)^2 + 0.5*lambda*(apply((w1_M1[aa_m,]^2 + w1_P1[aa_p,]^2),1,sum) ))  
      #求出损失函数
      #其实之所以这样计算是为了降低计算量，如果每一次都uv相乘，计算量太大
      #Compute Gradients %%%%%%%%%%%%%%%%%%%迭代 
      IO =2*(pred_out - rating)
      kkkk=num_feat-1
      for(kkk in 1:kkkk){
        IO =cbind(IO,2*(pred_out - rating))
      }
      #IO = repmat(2*(pred_out - rating),1,num_feat)
      #将损失矩阵的二倍，复制10列
      Ix_m=IO*w1_P1[aa_p,] + lambda*w1_M1[aa_m,]#损失*U-lambda*V 就是更新规则
      Ix_p=IO*w1_M1[aa_m,] + lambda*w1_P1[aa_p,]#损失*V-lambda*U
      #还是不太懂他为啥能batch后这么计算 而且矩阵形式本来就更复杂
      dw1_M1 = matrix(rep(0,num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T)
      #生成全零movie特征矩阵
      dw1_P1 = matrix(rep(0,num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T)#生成全零用户特征矩阵
      
      for(ii in 1:N){#迭代一万次 每一行一行来 得到更新矩阵
        dw1_M1[aa_m[ii],]=  dw1_M1[aa_m[ii],] +  Ix_m[ii,]
        dw1_P1[aa_p[ii],]=  dw1_P1[aa_p[ii],] +  Ix_p[ii,]
      }
      
      # Update movie and user features %%%%%%%%%%%
      #真正开始更新 新的矩阵=过去矩阵+学习率*导数
      #全零特征矩阵=上次的得到的矩阵*动量+学习率*导数/1w
      w1_M1_inc =  epsilon*dw1_M1/N
      w1_M1 =  w1_M1 - w1_M1_inc#原矩阵-负导数*学习率
      w1_P1_inc = epsilon*dw1_P1/N
      w1_P1 =  w1_P1 - w1_P1_inc
      
      
    }
    
    
    #此时所有(9轮)batch结束 
    #现在已经得到了此轮epoch后的一组U和V
    
    # Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
    pred_out= apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum)
    #其实就是U*V 但为什么只预测最后1w个？
    f_s=sum((pred_out - rating)^2 + 0.5*lambda*(apply((w1_M1[aa_m,]^2 + w1_P1[aa_p,]^2),1,sum) ))  
    
    err_train3[epoch] = sqrt(f_s/N)
    
    #计算平均loss的root（除了1w） 可能因为计算量吧，只计算了最后1w个loss 
    #Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
    NN=pairs_pr#验证集长度
    
    aa_p = probe_vec[,1]#读取验证集的user、movie、rating
    aa_m = probe_vec[,2]
    rating = probe_vec[,3]
    
    
    pred_out =apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum) + mean_rating#预测结果加上mean才行
    pred_out[pred_out>5]=5
    pred_out[pred_out<1]=1
    #使得预测结果超过评分区间的值依旧掉在区间内
    
    err_valid3[epoch]= sqrt(sum((pred_out- rating)^2)/NN)
    #print(paste('epoch',epoch,'Train RMSE',signif(err_train3[epoch], 4),'Test RMSE',signif(err_valid3[epoch], 4)))
    ################################输出到屏幕
    
    # err_random[epoch]=sqrt(sum((runif(10001,1,5) - rating)^2)/NN)
    
  }
  
  return(list(err1=err_train3,  err2= err_valid3))
}

