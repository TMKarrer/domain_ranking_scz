# Visualization of the pipeline-specific feature importance of the meta-priors.

# libraries
library(ggplot2)
library(dplyr)
library(magrittr)
library(viridis)
library(ggjoy)

project_dir = '/Users/Teresa/sciebo/Meta_Priors_teresa'
modality_list = c('vbm', 'rs', 'vbm_rs')
taxonomy = c('BD', 'PC')
methods_list = c('all_methods')
methods <- c('ha_pc', 'md_pc', 'ts_pc', 'kmeans', 'ward', 'spectral', 'pca', 'ica', 'sparse_pca')
methods_labels <- c('ha_pc', 'md_pc', 'ts_pc', 'K-means', 'ward', 'spectral', 'PCA', 'ICA', 'sparse PCA')
k_list <- c(5000, 5000, 5000, 100, 100, 100, 100, 100, 100)

for (modality in modality_list){
  print(modality)
  for (i in taxonomy) {
    print(i)
    if (i == 'BD'){
      name = 'Mental domains'
      domain_breaks = c(1.0, 10.0, 20.0, 30.0)
      w = 9
      h = 9
    } else if (i == 'PC'){
      name = 'Experimental tasks'
      domain_breaks = c(1.0, 10.0, 20.0, 30.0, 40.0, 50.0)
      w = 11
      h = 15
    }
    
    setwd(paste(project_dir, "models/LogReg_RF", modality, "_meta_ranking", "all_methods", .Platform$file.sep, sep='/'))
    data_mean <- read.csv(file=paste(i, '_', modality, '_all_methods_BT_data_mean.csv', sep='')) %>% select(-X)
    data_mean <- data_mean[order(data_mean$means),] 
    rank_data_mean <- read.csv(file=paste(i, '_', modality, '_all_methods_rank_BT_data_mean.csv', sep='')) %>% select(-X)
    rank_data_mean <- rank_data_mean[order(rank_data_mean$means),] 
    
    for (modality in modality_list){
      for (i in taxonomy) {
        k_idx <- 0
        data <- data.frame()
        rank_data <- data.frame()
        for (method in methods) {
          k_idx <- k_idx + 1
          setwd(paste(project_dir, "/models/LogReg_RF/", modality, '/', method, .Platform$file.sep, sep=''))
          pipeline_data <- read.csv(file=paste(i, '_', method, '_', k_list[k_idx], '_BT_data_mean.csv', sep=''))
          pipeline_data <- pipeline_data %>% dplyr:: select(-X)
          pipeline_data$method <- rep(method, nrow(pipeline_data))
          data <- rbind(data, pipeline_data)
          
          pipeline_rank_data <- data.frame(pipeline_data$priors, rank(-pipeline_data$means))
          colnames(pipeline_rank_data) <- c("priors", "means")
          pipeline_rank_data$method <- rep(method, nrow(pipeline_rank_data))
          rank_data <- rbind(rank_data, pipeline_rank_data)
        }
      }
    }   
    
    setwd(paste(project_dir, "models/LogReg_RF", modality, "_meta_ranking", "all_methods", .Platform$file.sep, sep='/'))
    rank_data <- group_by(rank_data, priors) %>%
      mutate(m=mean(rank_data)) %>%
      arrange(m) %>%
      ungroup() %>%
      mutate(priors=factor(priors, levels = rank_data_mean$priors[order(rank_data_mean$means, decreasing=TRUE)] %>% unique))
    
    ggplot() +
      geom_point(data=rank_data, aes(y=priors, x=means, color=method), shape=16, size=2.5, alpha=0.7) +
      geom_point(data=rank_data_mean, aes(y=priors, x=means, color='Mean'), shape=18, size=3) +
      ylab(name) + xlab("Discriminabilty rank") +
      theme(panel.background=element_rect(fill='white'),
            axis.ticks = element_blank(), axis.line = element_line(color="grey", size = 0.1),
            text = element_text(size=17), axis.text=element_text(size=14)) +
      scale_x_continuous(breaks=domain_breaks) +
      scale_color_manual(name="Brain sampling\nstrategy",
                          breaks=c('ha_pc', 'md_pc', 'ts_pc', 'kmeans', 'ward', 'spectral', 'pca', 'ica', 'sparse_pca', 'Mean'),
                          labels=c('ha_pc', 'md_pc', 'ts_pc', 'K-means', 'ward', 'spectral', 'PCA', 'ICA', 'sparse PCA', 'Mean'),
                          values=c('ha_pc' = '#FFFF00', 'md_pc' = '#FFCC00', 'ts_pc' = '#FF9900', 'kmeans'= '#00CC33', 'ward' = '#009933', 'spectral' = '#006633', 
                                   'pca' = '#66CCCC', 'ica'= '#339999', 'sparse_pca' = '#003399', 'Mean' = 'black')) +
      ggsave(file=paste(i, '_', modality, '_all_methods_rank_BT_feature_importance_joyplot_pipeline_specific.png', sep=''), width = w, height = h)
  }
}

