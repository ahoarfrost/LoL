library(RColorBrewer)
library(ggplot2)

df = read.csv('Embs_SubsetEvenEnv_Exploded.csv')
Y = df[,colnames(df)[6:109]]
#xnam <- paste("df$X", 0:99, sep="")
Y = cbind(df$emb0,  df$emb1,  df$emb2,  df$emb3,  df$emb4,  df$emb5,  df$emb6,  df$emb7,  df$emb8,  df$emb9, df$emb10, df$emb11, df$emb12, df$emb13, df$emb14, df$emb15, df$emb16, df$emb17, df$emb18, df$emb19, df$emb20, df$emb21, df$emb22, df$emb23, df$emb24, df$emb25, df$emb26, df$emb27, df$emb28, df$emb29, df$emb30, df$emb31, df$emb32, df$emb33, df$emb34, df$emb35, df$emb36, df$emb37, df$emb38, df$emb39, df$emb40, df$emb41, df$emb42, df$emb43, df$emb44, df$emb45, df$emb46, df$emb47, df$emb48, df$emb49, df$emb50, df$emb51, df$emb52, df$emb53, df$emb54, df$emb55, df$emb56, df$emb57, df$emb58, df$emb59, df$emb60, df$emb61, df$emb62, df$emb63, df$emb64, df$emb65, df$emb66, df$emb67, df$emb68, df$emb69, df$emb70, df$emb71, df$emb72, df$emb73, df$emb74, df$emb75, df$emb76, df$emb77, df$emb78, df$emb79, df$emb80, df$emb81, df$emb82, df$emb83, df$emb84, df$emb85, df$emb86, df$emb87, df$emb88, df$emb89, df$emb90, df$emb91, df$emb92, df$emb93, df$emb94, df$emb95, df$emb96, df$emb97, df$emb98, df$emb99,df$emb100,df$emb101,df$emb102,df$emb103)
fit_env <- manova(Y ~ df$metaseek_env_package)
fit_anno <- manova(Y ~ df$annotation)
#fit_both = manova(Y ~ df$annotation*df$metaseek_env_package)
print(summary(fit_env))
print(summary(fit_anno))


datatest = read.csv("datatest_results3.csv")
datatest$max_seqs = as.factor(datatest$max_seqs)
datatest$train_size = factor(datatest$train_size,levels =c("38k","200k","2M","10M","20M","class (6.6M)","order (18M)"))
seq_colors = brewer.pal(7,'PuBu')
seq_colors = seq_colors[3:6]
other_colors = c("palevioletred","chartreuse3")

ggplot(datatest,aes(epoch,accuracy)) + geom_point(aes(color=train_size),size=3) + 
  geom_path(aes(color=train_size),size=1) + 
  scale_y_continuous('accuracy',breaks=c(0.30,0.32,0.34,0.36,0.37,0.38,0.39,0.40),limits=c(0.30,0.42)) + 
  scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7),limits=c(0,7)) + 
  scale_color_manual(values=c(seq_colors,other_colors)) + 
  theme_bw() + theme(panel.grid.minor = element_blank())
#ggplot(datatest) + geom_point(aes(x=epoch,y=train_loss,color=train_size),size=3) + geom_path(aes(x=epoch,y=train_loss,color=train_size),size=1) + geom_point(aes(x=epoch,y=valid_loss,color=train_size),size=3) + geom_path(aes(x=epoch,y=valid_loss,color=train_size),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9,10,11,12),limits=c(0,12)) + scale_color_manual(values=c(seq_colors,other_colors))

'''
tune = read.csv("wd_moms_tune_results.csv")
tune$wd = as.factor(tune$wd)
tune$moms = as.factor(tune$moms)
tune$wd_moms = as.factor(tune$wd_moms)
ggplot(tune,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))

m098 = tune[tune$moms=='(0.98,0.9)',]
ggplot(m098,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))
m095 = tune[tune$moms=='(0.95,0.85)',]
ggplot(m095,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))
m09 = tune[tune$moms=='(0.9,0.8)',]
ggplot(m09,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))
w2 = tune[tune$wd=='0.01',]
ggplot(w2,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))
w3 = tune[tune$wd=='0.001',]
ggplot(w3,aes(epoch,accuracy)) + geom_point(aes(color=wd_moms),size=3) + geom_path(aes(color=wd_moms),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))

drop = read.csv('tune_drop_results.csv')
drop$drop_mult = factor(drop$drop_mult, levels=c('0.01','0.05','0.1','0.2'))
ggplot(drop,aes(epoch,accuracy)) + geom_point(aes(color=drop_mult),size=3) + geom_path(aes(color=drop_mult),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))
ggplot(drop) + geom_point(aes(x=epoch,y=train_loss,color=drop_mult),size=3) + geom_path(aes(x=epoch,y=train_loss,color=drop_mult),size=1) + geom_point(aes(x=epoch,y=valid_loss,color=drop_mult),size=3) + geom_path(aes(x=epoch,y=valid_loss,color=drop_mult),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9)) 
drop$loss_diff = drop$train_loss-drop$valid_loss
ggplot(drop,aes(epoch,loss_diff)) + geom_point(aes(color=drop_mult),size=3) + geom_path(aes(color=drop_mult),size=1) + scale_x_continuous('epoch',breaks=c(0,1,2,3,4,5,6,7,8,9),limits=c(0,9))


#pca for the embedding vectors of the 47k subset sequences
df.pca <- prcomp(df[,c(2:101)])
summary(df.pca)
library(ggbiplot)
ggbiplot(df.pca)
ggbiplot(df.pca, var.axes=FALSE)
ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.3)
ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.4, ellipse=TRUE) + scale_color_manual(values=brewer.pal(10,"Set3"))
#manual choices - not good
#colors = c("seagreen","gold","purple","red","lavenderblush3","yellow","chartreuse2","violetred","blue","darkmagenta")
#set2 plus 2 extra
#colors = c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#BC80BD","#B3B3B3","#80B1D3")
#set1 plus an extra
colors = c("#E41A1C", "blue", "#4DAF4A", "#984EA3", "#FF7F00", "gold", "#A65628", "#F781BF", "#999999","turquoise")
ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.3, choices=c(1,2)) + scale_color_manual(values=colors) + theme_bw()

#ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.3, ellipse=TRUE) + scale_color_manual(values=colors) + theme_bw()
ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.3, choices=c(3,4)) + scale_color_manual(values=colors) + theme_bw()
ggbiplot(df.pca, var.axes=FALSE, groups=df$metaseek_env_package, alpha=0.3, choices=c(5,6)) + scale_color_manual(values=colors) + theme_bw()

#pca <- prcomp(df[,c(2:101)], rank=2)
#ggbiplot(pca)

#t-sne for the embedding vectors
#library(vegan)
#library(M3C)
#duplicated <- duplicated(df[,c(2:101)])
#tidy <- df[!duplicated,]
#df_t <- t(tidy[,c(2:101)])
#labels <- tidy$metaseek_env_package
#tsne(df_t,labels=labels, colvec=colors, alpha=0.3)
'''

library(Rtsne)
library(inlmisc)
cols = GetColors(n=10)
#cols - light purple, purple, dark purple, blue, turquoise, mint, yellow, orange, red, brown
#["#E7EBFA", "#B997C6", "#824D99", "#4E79C4", "#57A2AC", "#7EB875", "#D0B440", "#E67F33", "#CE2220", "#521913"]

cols2 = c("#CE2220","#984EA3", "#521913", "#F781BF","#999999", "#228C22", "blue", "#FF7F00", "#D0B440", "#7DCACC")
colors = c("#E41A1C", "blue", "#4DAF4A", "#984EA3", "#FF7F00", "gold", "#A65628", "#F781BF", "#999999","turquoise")
tsne <- Rtsne(df[,c(6:109)], check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)
#save tsne results as dataframe
tsne_valid = data.frame(tsne1=tsne$Y[,1], tsne2=tsne$Y[,2], metaseek_env_package=df$metaseek_env_package)
write.csv(tsne_valid, "tsne_results_valid.csv")

df$metaseek_env_package <- factor(df$metaseek_env_package, 
                 levels=c("miscellaneous","host-associated","wastewater/sludge","human-gut","microbial mat/biofilm", 
                          "water","soil","sediment","plant-associated","built environment") )
cols3 = c("#999999", "#984EA3", "#D0B440","#521913", "#F781BF","#7DCACC","#FF7F00","blue", "#228C22", "#CE2220")

ggplot(df,aes(x=tsne$Y[,1], y=tsne$Y[,2], color=metaseek_env_package)) + 
  geom_point(alpha=0.1,size=1.5) + scale_color_manual(values=cols3) + 
  theme_bw() + xlab('t-SNE1') + ylab('t-SNE2') + 
  guides(colour = guide_legend(override.aes = list(alpha = 1), title="environmental package"))

even = read.csv('Embs_SubsetEven20k.csv')
#do tsne for even env classes (20k each)
even_tsne <- Rtsne(even[,c(7:ncol(even))], check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)
#save 
even_tsne_df = data.frame(tsne1=even_tsne$Y[,1], tsne2=even_tsne$Y[,2], metaseek_env_package=even$metaseek_env_package)
write.csv(even_tsne_df, "tsne_results_even20k.csv")

even$metaseek_env_package <- factor(even$metaseek_env_package, 
                                  levels=c("miscellaneous","wastewater/sludge","microbial mat/biofilm","human-gut", 
                                           "water","soil","sediment","plant-associated","host-associated","built environment") )
cols3 = c("#999999","#D0B440", "#F781BF","#521913", "#7DCACC","#FF7F00","blue", "#228C22", "#824D99", "#CE2220")

ggplot(even,aes(x=even_tsne$Y[,1], y=even_tsne$Y[,2], color=metaseek_env_package)) + 
  geom_point(alpha=0.1,size=1.5) + scale_color_manual(values=cols3) + 
  theme_bw() + xlab('t-SNE1') + ylab('t-SNE2') + 
  guides(colour = guide_legend(override.aes = list(alpha = 1), title="environmental package",size=36))


#df_mat = as.matrix(df[,c(2:101)]) #embedding info
#set.seed(246)
#nmds = metaMDS(df_mat,distance="bray")

