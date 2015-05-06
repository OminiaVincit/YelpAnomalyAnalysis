source('./func_utils.R')

# Validation

# Read data
features_data <- read.table('../features_votes_10_topic_only.txt', sep = ' ', header = F)

validate <- function(dat){
  num_trial = 20
  for (rate in c(0.25, 0.50, 0.75, 1.0)){
    train_rate = rate * 0.5
    lm_res = rep(0, num_trial)
    glm_res = rep(0, num_trial)
    #base_line = rep(0, num_trial)
    
    for (i in 1:num_trial){
      mydata = dat[sample(nrow(dat)), ]
      weights_cl = ncol(mydata)# The last column is weights column
      target_cl = weights_cl - 1
      mydata = makeTest(mydata, 0.5 - train_rate, 0.5)
      data_ls = dataPrep(mydata, c(target_cl, weights_cl))
      X_tr = data_ls$training
      X_ts = data_ls$testing
      
      features_col = 1:(target_cl-1)
      #lm_coeff = glmGauss(X_tr, features_col, target_cl)
      #lm_res[i] = compute_RMSE(X_ts, features_col, lm_coeff, target_cl, type='lm')
      
      #lm_fit = glmGauss(X_tr, features_col, target_cl)
      #lm_res[i] = AIC(lm_fit, k=log(length(X_tr[,1])))
      
      glm_coeff = glmBinom(X_tr, features_col, target_cl, weights_cl)
      glm_res[i] = compute_RMSE(X_ts, features_col, glm_coeff, target_cl, type='glm')
      
      #base_line[i] = baseline_RMSE(X_tr[, target_cl], X_ts[, target_cl])
      
      #glm_fit = glmBinom(X_tr, features_col, target_cl, weights_cl)
      #glm_res[i] = AIC(glm_fit, k=log(length(X_tr[,1])))
      
    }
    
    #lm_mean = median(lm_res)
    #lm_sd = sd(lm_res)
    
    glm_mean = median(glm_res)
    glm_sd = sd(glm_res)
    
    #bs_mean = mean(base_line)
    #bs_sd = sd(base_line)
    
    #print (c('Baseline: ', rate, bs_mean, bs_sd, base_line))
    #print (c('Lm: ', rate, lm_mean, lm_sd))
    print (c('Glm: ', rate, glm_mean, glm_sd))
  }
}
print ('features_data')
validate(features_data)

