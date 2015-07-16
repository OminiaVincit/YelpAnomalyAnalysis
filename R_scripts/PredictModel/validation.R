source('./func_utils.R')

# Validation

# Read data
featuresDat <- read.table('../Data/tripadvisor_all_features.txt', sep = ' ', header = F)

topics_ncol <- 50
util_ncol <- 3

validate <- function(dat){
  num_trial = 20
  for (rate in c(0.25, 0.50, 0.75, 1.0)){
    train_rate = rate * 0.5
    lm_res = rep(0, num_trial)
    glm_res = rep(0, num_trial)
    
    for (i in 1:num_trial){
      mydata = dat[sample(nrow(dat)), ]
      target_cl = ncol(mydata) # The last column is target column
      weights_cl = target_cl - 1
      
      # Only topics
      # features_col = (target_cl - util_ncol - topics_ncol+1):(target_cl - util_ncol)
      
      # Both topics and text
      features_col = 1:(target_cl - util_ncol)
      
      # Only text
      # features_col = 1:(target_cl - util_ncol - topics_ncol)
      
      mydata = makeTest(mydata, 0.5 - train_rate, 0.5)
      data_ls = dataPrep(mydata, features_col)
      X_tr = data_ls$training
      X_ts = data_ls$testing
      
      lm_coeff = glmGauss(X_tr, features_col, target_cl)
      lm_res[i] = compute_RMSE(X_ts, features_col, lm_coeff, target_cl, type='lm')
      
      #lm_fit = glmGauss(X_tr, features_col, target_cl)
      #lm_res[i] = AIC(lm_fit, k=log(length(X_tr[,1])))
      
      glm_coeff = glmBinom(X_tr, features_col, target_cl, weights_cl)
      glm_res[i] = compute_RMSE(X_ts, features_col, glm_coeff, target_cl, type='glm')
      
      #glm_fit = glmBinom(X_tr, features_col, target_cl, weights_cl)
      #glm_res[i] = AIC(glm_fit, k=log(length(X_tr[,1])))
      
    }
    
    lm_median = median(lm_res)
    lm_sd = sd(lm_res)
    
    glm_median = median(glm_res)
    glm_sd = sd(glm_res)
    
    print (c('Lm: ', rate, lm_median, lm_sd))
    print (c('Glm: ', rate, glm_median, glm_sd))
  }
}
print ('features_data')
validate(featuresDat)

