###
# Plot results for paper
###
library(ggplot2)

# Regression result
mae <- read.table('../Results/tripadvisor_predict_64_features_20150717.txt', sep = ',', header = F, skip=1)
df1 <- data.frame(percent=mae$V1, li_text=mae$V2, li_top=mae$V3, li_text_top=mae$V4, 
                  log_text=mae$V5, log_top=mae$V6, log_text_top=mae$V7)
g1 <- ggplot(
  df1,
  aes(
    x = percent, 
    y = MAE, 
    color = "Features Design"
  )
)
g1 <- g1 + geom_line(aes(y=li_text, color="TRIPADVISOR-Linear-Text-Only"), size=1.5, linetype=3) + 
  geom_point(aes(y=li_text, color="TRIPADVISOR-Linear-Text-Only"), size=4, shape=19) +
  geom_line(aes(y=li_top, col="TRIPADVISOR-Linear-Topics-Only"), size=1.5, linetype=4) + 
  geom_point(aes(y=li_top, col="TRIPADVISOR-Linear-Topics-Only"), size=4, shape=15) +
  geom_line(aes(y=li_text_top, col="TRIPADVISOR-Linear-Text-Topics"), size=1.5, linetype=1) + 
  geom_point(aes(y=li_text_top, col="TRIPADVISOR-Linear-Text-Topics"), size=6, shape=18) +
  geom_line(aes(y=log_text, col="TRIPADVISOR-Logistic-Text-Only"), size=1.5, linetype=3) + 
  geom_point(aes(y=log_text, col="TRIPADVISOR-Logistic-Text-Only"), size=4, shape=19) +
  geom_line(aes(y=log_top, col="TRIPADVISOR-Logistic-Topics-Only"), size=1.5, linetype=4) + 
  geom_point(aes(y=log_top, col="TRIPADVISOR-Logistic-Topics-Only"), size=4, shape=15) + 
  geom_line(aes(y=log_text_top, col="TRIPADVISOR-Logistic-Text-Topics"), size=1.5, linetype=1) +
  geom_point(aes(y=log_text_top, col="TRIPADVISOR-Logistic-Text-Topics"), size=6, shape=18) +
  scale_color_manual(values=c("TRIPADVISOR-Linear-Text-Only"="blue4", 
                              "TRIPADVISOR-Linear-Topics-Only"="blue4", 
                              "TRIPADVISOR-Linear-Text-Topics"="blue4",
                              "TRIPADVISOR-Logistic-Text-Only"="red3",
                              "TRIPADVISOR-Logistic-Topics-Only"="red3",
                              "TRIPADVISOR-Logistic-Text-Topics"="red3" )) + 
  theme(legend.direction="horizontal", legend.position="top")+
  xlab("Training data percentage (of 50% dataset)") +
  ylab("Test MAE Value")
g1