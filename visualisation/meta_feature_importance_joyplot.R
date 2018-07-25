# Visualization of the average feature importance of the meta-priors.

# libraries
library(ggplot2)
library(dplyr)
library(magrittr)
library(viridis)
library(ggjoy)

project_dir = 'scz_ranking_project'
modality_list = c('vbm', 'rs', 'vbm_rs')
taxonomy = c('BD', 'PC')
method_list = c('all_methods')

for (modality in modality_list){
  print(modality)
  for (met_i in seq(length(method_list))) {
    method = method_list[met_i]
    print(method)
    for (i in taxonomy) {
      setwd(paste(project_dir, "models/LogReg_RF", modality, "_meta_ranking", method, .Platform$file.sep, sep='/'))
      
      rank_data_mean <- read.csv(file=paste(i, '_', modality, '_', method, "_rank_BT_data_mean.csv", sep='')) %>% select(-X)
      rank_data_ci <- read.csv(file=paste(i, '_', modality, '_', method, "_rank_BT_data_ci.csv", sep='')) %>% select(-X)
      rank_joy_data <- read.csv(file=paste(i, '_', modality, '_', method, "_rank_BT_data_joy.csv", sep=''))  %>% select(-X)
      
      rank_data_ci %<>% merge(rank_data_mean, by="priors")
      rank_data_ci$priors <- factor(rank_data_ci$priors, levels = rank_data_ci$priors[order(rank_data_ci$means, decreasing=TRUE)] %>% unique)
      
      rank_joy_data <- group_by(rank_joy_data, priors) %>%
        mutate(m=mean(data)) %>%
        arrange(m) %>%
        ungroup() %>%
        mutate(priors=factor(priors, levels = rank_data_ci$priors[order(rank_data_ci$means, decreasing=TRUE)] %>% unique))
      
      if (i == 'BD'){
        name = 'Mental domains'
        w = 7
        h = 9
        domain_breaks = c(1.0, 10.0, 20.0, 30.0)
      } else if (i == 'PC'){
        name = 'Experimental tasks'
        w = 7
        h = 12
        domain_breaks = c(1.0, 10.0, 20.0, 30.0, 40.0, 50.0)
      }
      
      ggplot() +
        scale_fill_viridis(discrete = TRUE) +
        geom_joy(data=rank_joy_data, aes(x=data, y=priors, height=..density.., fill = priors), alpha=0.8, col = "black", scale = 2.4, show.legend = F) + 
        geom_line(data=rank_data_ci %>% mutate(priors = priors %>% as.factor()), aes(y=priors, x=values, group=priors, color='deeppink1')) +
        geom_point(data=rank_data_ci %>% mutate(priors = priors %>% as.factor()), aes(y=priors, x=means, group=priors, color='deeppink1'), shape=18, size=2) +
        ylab(name) + xlab("Discriminabilty of schizophrenia") +
        theme(legend.position='none',
              panel.background=element_rect(fill='white'),
              axis.ticks = element_blank(), axis.line = element_line(color="grey", size = 0.1),
              text = element_text(size=17), axis.text=element_text(size=14)) +
        scale_x_continuous(breaks=domain_breaks) +
        ggsave(file=paste(i, '_', modality, '_', method, "_rank_BT_feature_importance_joyplot.png", sep=''), width = w, height = h)
      
    }
  }
}
