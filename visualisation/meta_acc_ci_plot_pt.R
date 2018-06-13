'''
Visualization of the permutation test.
'''

# libraries
library(ggplot2)
library(dplyr)
library(magrittr)
library(xlsx)
library(viridis)

project_dir = 'scz_ranking_project'
taxonomy = c('BD','PC')
modality_list = c('vbm', 'rs', 'vbm_rs')

for (modality in modality_list){
  print(modality)
  for (i in taxonomy) {
    setwd(paste(project_dir, "/models/LogReg_RF/", modality, "/_meta_ranking", .Platform$file.sep, sep=''))
    data_ci <- read.csv(file=paste(i, '_', "acc_ci_pt.csv", sep='')) %>% select(-X)
    data_ci$methods <- factor(data_ci$methods, levels = data_ci$methods[order(data_ci$means)] %>% unique)
    data_distr <- read.csv(file=paste(i, '_', "acc_distribution_pt.csv", sep='')) %>% select(-X)
    data_distr <- group_by(data_distr, methods) %>%
      ungroup() %>%
      mutate(methods=factor(methods, levels = data_ci$methods[order(data_ci$means)] %>% unique))  
    data_ci$methods <- ifelse(data_ci$methods=="ha_pc", "1. peaks", 
                       ifelse(data_ci$methods == "md_pc", "2. peaks",
                       ifelse(data_ci$methods == "ts_pc", "3. peaks",
                       ifelse(data_ci$methods == "kmeans", "4. regions",
                       ifelse(data_ci$methods == "ward", "5. regions", 
                       ifelse(data_ci$methods == "spectral", "6. regions",
                       ifelse(data_ci$methods == "pca", "7. networks",
                       ifelse(data_ci$methods == "sparse_pca", "8. networks",  
                       ifelse(data_ci$methods == "ica", "9. networks", '???')))))))))
     
    data_distr$methods <- ifelse(data_distr$methods=="ha_pc", "1. peaks", 
                          ifelse(data_distr$methods == "md_pc", "2. peaks",
                          ifelse(data_distr$methods == "ts_pc", "3. peaks",
                          ifelse(data_distr$methods == "kmeans", "4. regions",
                          ifelse(data_distr$methods == "ward", "5. regions",
                          ifelse(data_distr$methods == "spectral", "6. regions",
                          ifelse(data_distr$methods == "pca", "7. networks",
                          ifelse(data_distr$methods == "sparse_pca", "8. networks",  
                          ifelse(data_distr$methods == "ica", "9. networks", '???')))))))))    
  
    ggplot() + 
      scale_color_viridis(discrete=TRUE) +
      scale_fill_viridis(discrete=TRUE) +
      stat_boxplot(geom ='errorbar', width=0.5, data=data_distr[data_distr$type=='perm_test',], aes(methods, values), color = 'lightblue4', lwd=1,  fill=NA, show.legend=FALSE, outlier.shape=NA, coef=1.0) +
      geom_boxplot(data=data_distr[data_distr$type=='perm_test',], aes(methods, values), color = 'lightblue4', lwd=1,  fill='white', show.legend=FALSE, outlier.shape=NA, coef=1.0) +    
      geom_jitter(data=data_distr[data_distr$type=='perm_test',], aes(methods, values, color = methods), alpha = 0.15) +
      geom_point(data=data_ci[data_ci$type=='result',], aes(methods, means), shape = 18, size = 6, color = 'magenta4') +
      theme(legend.position='none',
            panel.background=element_rect(fill='white'),
            axis.ticks = element_blank(), axis.line = element_line(color="grey", size = 0.1),
            text = element_text(size=20), axis.text=element_text(size=17),
            strip.background = element_rect(fill="white"),
            axis.line.x = element_blank(), 
            axis.text.x = element_text(size = 17, angle = 60, hjust=1)) +
      xlab('Pipelines') + ylab("Prediction accuracy") +
      scale_y_continuous(labels = scales::percent, breaks = c(0.25, 0.5, 0.75), limits = c(0.2, 0.8)) + 
    ggsave(file=paste(i, '_', modality, "_meta_acc_ci_plot_pt.png", sep=''), dpi=500)
    
  }
}