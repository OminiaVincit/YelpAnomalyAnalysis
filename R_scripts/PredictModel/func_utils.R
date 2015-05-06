# Divide to train, cross, and test set
makeTest <- function(srcdat, cross_r, test_r){
  mydata = srcdat
  mydata$Case = 'Train'
  n_rw = nrow(mydata)
  num_cross = as.integer(n_rw * cross_r)
  num_test = as.integer(n_rw * test_r)
  if (num_cross > 0) {
    mydata$Case[1:num_cross] = 'Cross'  
  }
  if (num_test > 0) {
    mydata$Case[(num_cross+1):(num_test+num_cross)] = 'Test'
  }
  return (mydata)
}

# Preprocessing for data
transform <- function(X, f_means, f_sds){
  nc = ncol(X)
  for (i in 1:nc){
    X[, i] <- (X[, i] - f_means[i])/f_sds[i]
  }
  return (X)
}

dataPrep <- function(mydata, exclude){
  training = subset(mydata, Case=='Train')
  training$Case = NULL

  crossing = subset(mydata, Case=='Cross')
  crossing$Case = NULL
  
  testing = subset(mydata, Case=='Test')
  testing$Case = NULL
  
  f_sds = apply(training, 2, sd)
  f_means = apply(training, 2, mean)
  
  for (exc in exclude){
    f_sds[exc] = 1
    f_means[exc] = 0
  }
  
  training = transform(training, f_means, f_sds)
  
  if (nrow(testing) > 0){
    testing = transform(testing, f_means, f_sds)
  }
  if (nrow(crossing) > 0){
    crossing = transform(crossing, f_means, f_sds)
  }
  
  newlist = list('training' = training, 'crossing' = crossing, 'testing' = testing)
  return (newlist)
}

linearModel <- function(X, features_col, target_cl){
  # Fitting linear regression
  xnam = paste("X$V", features_col, sep="")
  ynam = paste("X$V", target_cl, sep="")
  fmla = as.formula(paste(ynam, " ~ ", paste(xnam, collapse= "+")))
  #lm_fit <- lm(X$V12 ~ X$V1 + X$V2 + X$V3 + X$V4 + X$V5 + X$V6 + X$V7 + X$V8 + X$V9 + X$V10 + X$V11, 
  #             data=X)
  lm_fit = lm(fmla, data=X)
  
#   summary(lm_fit)
#   coefficients(lm_fit) # model coefficients
#   confint(lm_fit, level=0.95) # CIs for model parameters 
#   fitted(lm_fit) # predicted values
#   rs <- residuals(lm_fit, type='response') # residuals
  #mean(rs*rs)/2.0
  
  #anova(lm_fit) # anova table 
  #vcov(lm_fit) # covariance matrix for model parameters 
  #influence(lm_fit) # regression diagnostics
  
  # diagnostic plots 
  #layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
  #plot(lm_fit)
  
  # K-fold cross-validation
  # library(DAAG)
  # cv.lm(df=X, lm_fit, m=3) # 3 fold cross-validation
  #return (coefficients(lm_fit))
  return (lm_fit)
}

glmGauss <- function(X, features_col, target_cl){
  xnam = paste("X$V", features_col, sep="")
  ynam = paste("X$V", target_cl, sep="")
  fmla = as.formula(paste(ynam, " ~ ", paste(xnam, collapse= "+")))
  glm_fit = glm(fmla, data=X, family=gaussian)
  #return (coefficients(glm_fit))
  return (glm_fit)
}

glmBinom <- function(X, features_col, target_cl, weights_cl){
  # Fitting by generalized model
  xnam = paste("X$V", features_col, sep="")
  ynam = paste("X$V", target_cl, sep="")
  #ynam = paste("X$V", c(target_cl, weights_cl), sep="")
  #ynam = paste(ynam, collapse=",")
  #ynam = paste("cbind(", ynam, ")")
  fmla = as.formula(paste(ynam, " ~ ", paste(xnam, collapse= "+")))
  #X[, target_cl] = floor(X[, target_cl] * X[, weights_cl])
  #X[, weights_cl] = X[, weights_cl] - X[, target_cl]
  #glm_fit <- glm(X$V12 ~ X$V1 + X$V2 + X$V3 + X$V4 + X$V5 + X$V6 + X$V7 + X$V8 + X$V9 + X$V10 + X$V11, 
  #               weights=X$V13, data=X, family=binomial(link=logit))
  
  glm_fit = glm(fmla, weights=X[, weights_cl], data=X, family=binomial(link=logit))
  #glm_fit = glm(fmla, data=X, family=binomial(link=logit))
  #return (coefficients(glm_fit))
  return (glm_fit)
}

glmPoisson <- function(X, features_col, target_cl){
  # Fitting by generalized model
  xnam = paste("X$V", features_col, sep="")
  ynam = paste("X$V", target_cl, sep="")
  fmla = as.formula(paste(ynam, " ~ ", paste(xnam, collapse= "+")))
  glm_fit = glm(fmla, data=X, family=poisson(link=logit))
  return (glm_fit)
}

compute_RMSE <- function(X, features_col, coeff, target_cl, type='lm'){
  # The index of target column
  target = X[, target_cl]
  predicted = coeff[1] + rep(0, length(target))
  for (i in features_col){
    predicted = predicted + X[, i] * coeff[i+1]
  }
  if (type=='glm'){
    predicted = 1.0 - 1.0 / (1.0 + exp(predicted))
  }
  diff = predicted - target
  #rmse = mean(diff*diff)/2.0
  rmse = mean(abs(diff))
  return (rmse)
}

baseline_RMSE <- function(cl_tr, cl_ts){
  # base line as the mean review quality in the training data
  qual = mean(cl_tr)
  diff = cl_ts - qual
  rmse = mean(diff*diff)/2.0
  return (rmse)
}

dat <- read.table('../features_votes_10_text_only.txt', sep = ' ', header = F)
mydata = dat[sample(nrow(dat)), ]
weights_cl = ncol(mydata)# The last column is weights column
target_cl = weights_cl - 1
mydata = makeTest(mydata, 0, 0)
data_ls = dataPrep(mydata, c(target_cl, weights_cl))
X_tr = data_ls$training
# X_ts = data_ls$testing
# 
features_col = 1:(target_cl-1)
#lm_fit = linearModel(X_tr, features_col, target_cl)
glm_fit = glmBinom(X_tr, features_col, target_cl, weights_cl)
#glm_res[i] = AIC(glm_fit, k=log(length(X_tr[,1])))
#layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
#plot(glm_fit)
df1 <- data.frame(fitted = fitted(glm_fit), votes=X_tr[, weights_cl], residuals=residuals(glm_fit, type='pearson') )

g1 <- ggplot(df1, aes(x = fitted, y = votes))
g1 <- g1 + geom_point(color = 'red')
g1 <- g1 + labs(x = 'Fitted values', y = 'Number of votes')

g2 <- ggplot(df1, aes(x = fitted, y = residuals))
g2 <- g2 + geom_point(color = 'blue')
g2 <- g2 + labs(x = 'Fitted values', y = 'Residuals')

multiplot(g1, g2, cols = 1)

