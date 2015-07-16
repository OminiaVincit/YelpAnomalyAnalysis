### Features selection 

library(ggplot2)

# corrplot: the library to compute the correlation matrix
library(corrplot)

# read the tab file using the read table function
datMy <- NULL
if (is.null(datMy)) {
  datMy <- read.table('./Data/movies_all_features.txt')
}

numCol = ncol(datMy)
# # scale all the featutures except the last 3 elements (for predictor)
# datMy.scale <- scale(datMy[1:(numCol-3)], center=TRUE, scale=TRUE)
# 
# # compute the correlation matrix
# corMatMy <- cor(datMy.scale)
# 
# # visualize the matrix, clustering features by correlation index
# corrplot(corMatMy, order='original')

# Apply correlation filter at 0.70,
# then we remove all the variable correlated with more 0.7.
# highlyCor <- findCorrelation(corMatMy, 0.70)
# datMyFiltered.scale <- datMy.scale[,-highlyCor]
# corMatMy <- cor(datMyFiltered.scale)
# corrplot(corMatMy, order = "hclust")

# Apply PCA

# PCA with function PCA
require(FactoMineR)

# scale all the features, ncp: number of dimensions kept in the results
pca <- PCA(datMy[1:(numCol-3)], scale.unit=TRUE, ncp=3, graph=T)

# Sort the variables most linked to each PC
# dimdesc(pca)