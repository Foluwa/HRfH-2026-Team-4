
############################
# QQPLOTs
############################

############################
# Import libraries
############################
library(ggplot2)
library(ggpubr)
library(tidyverse)

############################
# Import dataset
############################
df <- read.csv("//uol.le.ac.uk/root/staff/home/a/ar671/Downloads/patients_df.csv")


############################
# Age
############################
hist_age <- df %>% ggplot(aes(x=age)) +
  geom_bar()

fig_age <- df %>% ggplot(aes(sample= age))+
  stat_qq(color = "red")+
  theme_classic()+
  labs(title = "Age")

fig_bmi <- df %>% ggplot(aes(sample= BMI))+
  stat_qq(color = "blue")+
  theme_classic()+
  labs(title = "BMI")

fig_max_steps <- df %>% ggplot(aes(sample= max_steps))+
  stat_qq(color = "green")+
  theme_classic()+
  labs(title = "Max steps")

fig_pain <- df %>% ggplot(aes(sample= baseline_pain))+
  stat_qq(color = "goldenrod1")+
  theme_classic()+
  labs(title = "Baseline Pain")

############################
# ALL Together
############################

fig <- ggarrange(fig_age, fig_bmi, fig_max_steps, fig_pain, 
                 nrow =2, ncol =2)

fig

ggsave("//uol.le.ac.uk/root/staff/home/a/ar671/Desktop Files/Useful Thing/qqplots.jpeg", fig)



