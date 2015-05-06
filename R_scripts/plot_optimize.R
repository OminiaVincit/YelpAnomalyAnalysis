library(ggplot2)
# Plot for optimization
source('multi_plot.r')

# Plot for iterations vs cost value
cost_1 = read.table('test_optimize_alpha_0.txt', sep = ' ', header = F)
p1 <- ggplot(cost_1, aes(x = V1, y = V2)) + 
  labs(x = 'Iterations', y = 'Cost') +
  geom_point(color = 'red') +
  ggtitle('Plot of number of iterations versus cost value (regularization term = 0)')

# Plot for downstep of learning rate
# p2 <- ggplot(cost_1, aes(x = V1, y = V3)) + 
#   labs(x = 'Iterations', y = 'Log(learning rate)') +
#   geom_point(color = 'red') +
#   scale_y_log10(limits = c(-1e8, 1e2)) + 
#   ggtitle('Plot of number of iterations versus Log(learning_rate)')

cost_2 = read.table('test_optimize_alpha_0.1.txt', sep = ' ', header = F)
p2 <- ggplot(cost_2, aes(x = V1, y = V2)) + 
  labs(x = 'Iterations', y = 'Cost') +
  geom_point(color = 'red') +
  ggtitle('Plot of number of iterations versus cost value (regularization term = 0.1)')

cost_3 = read.table('test_optimize_alpha_1.txt', sep = ' ', header = F)
p3 <- ggplot(cost_3, aes(x = V1, y = V2)) + 
  labs(x = 'Iterations', y = 'Cost') +
  geom_point(color = 'red') +
  ggtitle('Plot of number of iterations versus cost value (regularization term = 1.0)')

cost_4 = read.table('test_optimize_alpha_10.txt', sep = ' ', header = F)
p4 <- ggplot(cost_4, aes(x = V1, y = V2)) + 
  labs(x = 'Iterations', y = 'Cost') +
  geom_point(color = 'red') +
  ggtitle('Plot of number of iterations versus cost value (regularization term = 10.0)')

multiplot(p1, p2, p3, p4, cols = 2)
