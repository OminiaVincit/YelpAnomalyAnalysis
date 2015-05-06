library(ggplot2)
# Residual histogram plot
source('multi_plot.r')

# Checking linear regression
err = read.table('votes_residual.txt', sep = ' ', header = F)
df1 <- data.frame(votes_residual = err$V1)
g1 <- ggplot(
  df1,
  aes(
    x = votes_residual
  )
)
g1 <- g1 + geom_histogram()

votes = read.table('votes_rate_from_10.txt', sep = ' ', header = F)
df2 <- data.frame(votes = votes$V1)
g2 <- ggplot(
  df2,
  aes(
    x = votes
  )
)
g2 <- g2 + geom_histogram(binwidth=5)

#rate = read.table('quality.txt', sep = ' ', header = F)
df3 <- data.frame(rate = votes$V2)
g3 <- ggplot(
  df3,
  aes(
    x = rate
  )
)
g3 <- g3 + geom_histogram()

rate_err = read.table('rate_residual.txt', sep = ' ', header = F)
df4 <- data.frame(rate_residuals = rate_err$V1)
g4 <- ggplot(
  df4,
  aes(
    x = rate_residuals
  )
)
g4 <- g4 + geom_histogram()
multiplot(g2, g1, g3, g4, cols=2)