# Plot information about yelp reviews data
library(ggplot2)

source('multi_plot.r')

# # Plot for log-log number of reviews vs number of businesses
# rv_bs = read.table('reviews_businesses.txt', sep = ' ', header = F)
# p1 <- ggplot(rv_bs, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of reviews)', y = 'Log(number of businesses)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8)) +
#   ggtitle('Log-log plot of number of reviews versus number of businesses')
# 
# # Plot for log-log number of reviews vs number of users
# rv_us= read.table('reviews_users.txt', sep = ' ', header = F)
# p2 <- ggplot(rv_us, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of reviews)', y = 'Log(number of users)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8)) +
#   ggtitle('Log-log plot of number of reviews versus number of users')
# 
# # Plot for log-log number of votes vs number of reviews
# rv_votes= read.table('reviews_votes.txt', sep = ' ', header = F)
# p3 <- ggplot(rv_votes, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of votes)', y = 'Log(number of reviews)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8))
#   # ggtitle('(a) Log-log plot of number of votes versus number of reviews')
# 
# # Plot for log-log number of helpful votes vs number of reviews
# rv_helpful= read.table('reviews_helpful.txt', sep = ' ', header = F)
# p4 <- ggplot(rv_helpful, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of helpful votes)', y = 'Log(number of reviews)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8))
#   # ggtitle('(b) Log-log plot of number of helpful votes versus number of reviews')
# 
# # Plot for log-log percentage of helpful votes vs number of reviews
# rv_helpful_pt= read.table('reviews_helpful_percentage.txt', sep = ' ', header = F)
# p5 <- ggplot(rv_helpful_pt, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(percent of helpful votes)', y = 'Log(number of reviews)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e2)) 
#   #ggtitle('Log-log plot of percent of useful votes versus number of reviews')
# 
# # Plot for number of helpful votes vs number of reviews
# p6 <- ggplot(rv_helpful_pt, aes(x = V1, y = V2)) + 
#   labs(x = 'Percent of helpful votes', y = 'Number of reviews') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of percent of useful votes versus number of reviews')
# 
# # Plot for log-log number of feedback votes vs number of users
# user_received_votes = read.table('users_votes.txt', sep = ' ', header = F)
# p7 <- ggplot(user_received_votes, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of feedback votes)', y = 'Log(number of users)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8)) +
#   ggtitle('Log-log plot of number of feedback votes versus number of users')
# 
# # Plot for log-log number of feedback helpful votes vs number of users
# user_received_helpful_votes = read.table('users_helpful_votes.txt', sep = ' ', header = F)
# p8 <- ggplot(user_received_helpful_votes, aes(x = V1, y = V2)) + 
#   labs(x = 'Log(number of useful feedback votes)', y = 'Log(number of users)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(1, 1e8)) + 
#   scale_x_log10(limits = c(1, 1e8)) +
#   ggtitle('Log-log plot of number of useful feedback votes versus number of users')
# 
# multiplot(p3, cols = 1)

# topics_feature = read.table('topics_deviation.txt', sep = ' ', header = F)
# p9 <- ggplot(topics_feature, aes(x = V1, y = V8)) + 
#   labs(x = 'Rate deviation', y = 'Quality') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of rate deviation versus quality of reviews')
# 
# p10 <- ggplot(topics_feature, aes(x = V2, y = V8)) + 
#   labs(x = 'Business average normally', y = 'Quality') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of business average normally versus quality of reviews')
# 
# p11 <- ggplot(topics_feature, aes(x = V3, y = V8)) + 
#   labs(x = 'Minor-major anomaly', y = 'Quality') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of business minor-major anomaly versus quality of reviews')
# 
# p12 <- ggplot(topics_feature, aes(x = V4, y = V8)) + 
#   labs(x = 'Major-minor anomaly', y = 'Quality') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of business major-minor anomaly versus quality of reviews')
# 
# multiplot(p9, p10, p11, p12, cols = 2)

# 
# rv_top_all = read.table('review_qdiff_topdiff_all.txt', sep = ' ', header = F)
# p13 <- ggplot(rv_top_all, aes(x = V1, y = V2)) + 
#   labs(x = 'Review quality difference', y = 'Topic distribution difference') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality difference versus Topic distribution difference')

# rv_top_bss = read.table('KL_diff.txt', sep = ' ', header = F)
# df1 <- data.frame(qdiff = rv_top_bss$V1, topdiff=rv_top_bss$V2)
# for (value in c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)){
#   df2 = subset(df1, abs(topdiff) < value)
#   df3 = subset(df1, abs(topdiff) >= value)
#   print (c(sd(df2$qdiff), sd(df3$qdiff)))
# }
value = 0.04
df2 <- subset(df1, abs(topdiff) < value)
df3 <- subset(df1, abs(topdiff) >= value)

g1 <- ggplot(
  df2,
  aes(
    x = qdiff
  )
)
#g1 <- g1 + geom_histogram(binwidth=0.01)
g1 <- g1 + geom_density()

# g2 <- ggplot(
#   df2,
#   aes(
#     x = topdiff
#   )
# )
# g2 <- g2 + geom_histogram()

g3 <- ggplot(
  df3,
  aes(
    x = qdiff
  )
)
#g3 <- g3 + geom_histogram(binwidth=0.01)
g3 <- g3 + geom_density()

# g4 <- ggplot(
#   df3,
#   aes(
#     x = topdiff
#   )
# )
# g4 <- g4 + geom_histogram()

# p14 <- ggplot(rv_top_bss, aes(x = V1, y = V2)) + 
#   labs(x = 'Review quality difference', y = 'Topic distribution difference') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality difference versus topic distribution difference')
# 
# p15 <- ggplot(rv_top_bss, aes(x = V1, y = V3)) + 
#   labs(x = 'Review quality difference', y = 'Topic distribution normally') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality difference versus topic distribution normally')
# 
# p16 <- ggplot(rv_top_bss, aes(x = V1, y = V4)) + 
#   labs(x = 'Review quality difference', y = 'Topic distribution anomally 1') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality difference versus topic distribution anomally 1')
# 
# p17 <- ggplot(rv_top_bss, aes(x = V1, y = V5)) + 
#   labs(x = 'Review quality difference', y = 'Topic distribution anomally 2') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality difference versus topic distribution anomally 2')

#multiplot(p14, p15, p16, p17, cols = 2)
multiplot(g1, g3, cols = 2)

# bss_uss_data <- read.table('./features_user_business_topics.txt', sep = ' ', header = F)
# p18 <- ggplot(bss_uss_data, aes(x = V19, y = V13)) + 
#    labs(x = 'Review quality', y = 'Business topics normally') +
#    geom_point(color = 'red') +
#    ggtitle('Plot of review quality versus business topic normally')
# 
# p19 <- ggplot(bss_uss_data, aes(x = V19, y = V14)) + 
#   labs(x = 'Review quality', y = 'Business topics anormally 1') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality versus business topic anormally 1')
# 
# p20 <- ggplot(bss_uss_data, aes(x = V19, y = V15)) + 
#   labs(x = 'Review quality', y = 'Business topics anormally 2') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality versus business topic anormally ')
# 
# p21 <- ggplot(bss_uss_data, aes(x = V19, y = V16)) + 
#   labs(x = 'Review quality', y = 'User topics normally') +
#   geom_point(color = 'red') +
#   ggtitle('Plot of review quality versus user topic normally')

#multiplot(p18, p19, p20, p21, cols = 2)

# topics_feature = read.table('topics_deviation.txt', sep = ' ', header = F)
# df1 <- data.frame(quality = topics_feature$V8, rate_dev=topics_feature$V1)
# for (value in 0:50){
#   value = value / 50
#   df2 = subset(df1, (abs(rate_dev) >= value) & (abs(rate_dev) < value + 0.02))
#   q <- df2$quality
#   print (c(value, length(q), median(q), mean(q), sd(q)))
# }
