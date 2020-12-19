library(ggplot2)
library(RColorBrewer)
result = read.csv("ParamSearch_KSDrop_results.csv")
result$ksize = as.factor(result$ksize)
result$stride = as.factor(result$stride)
result$drop_mult = as.factor(result$drop_mult)
p <- ggplot(result, aes(x=ksize,y=acc2)) + geom_boxplot()
p + geom_dotplot(binaxis='y', stackdir='center', dotsize=1, aes(fill=stride),alpha=0.5)

stride1 = result[result$stride==1,]
s <- ggplot(stride1, aes(x=ksize,y=acc2)) + geom_boxplot()
s + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.5, aes(fill=drop_mult),alpha=0.5) + coord_cartesian(ylim=c(0.355,0.375))

stride3 <- result[result$stride==3,]
t <- ggplot(stride3, aes(x=ksize,y=acc1)) + geom_boxplot()
t + geom_dotplot(binaxis='y', stackdir='center', dotsize=1, aes(fill=drop_mult),alpha=0.5)

drop = result[result$drop_mult==0.1,]
drop$k_s = paste(drop$k,rep('.',nrow(drop)),drop$stride,sep='')
drop$k_s = factor(drop$k_s, levels=c('1.1','3.1','6.1','9.1','1.3','3.3','6.3'))
d <- ggplot(drop, aes(x=k_s,y=acc2)) + geom_boxplot()
d + geom_dotplot(binaxis='y', stackdir='center', dotsize=1, aes(fill=ksize),alpha=0.5)

pal = brewer.pal(n=10,"Paired")
embbptt = read.csv("EmbBpttResults.csv")
embbptt$bptt = as.factor(embbptt$bptt)
ggplot(embbptt, aes(x=emb,y=acc)) + geom_point(aes(color=bptt)) + geom_line(aes(color=bptt)) + coord_cartesian(ylim=c(0.37,0.38))
embbptt = read.csv("EmbBpttResults.csv")
embbptt$emb = as.factor(embbptt$emb)
ggplot(embbptt, aes(x=bptt,y=acc)) + geom_point(aes(color=emb),alpha=0.7) + geom_line(aes(color=emb)) + coord_cartesian(ylim=c(0.37,0.38)) + scale_color_manual(values=pal) + theme_bw()

ggplot(embbptt, aes(x=bptt,y=emb)) + geom_point(aes(color=acc)) + scale_color_gradient(low='white',high='black',limits=c(0.37,0.38),na.value="white")
ggplot(embbptt, aes(x=bptt,y=emb)) + geom_point(aes(color=acc,size=acc)) + scale_color_gradient(low='white',high='red',limits=c(0.372,0.378),na.value="white") + scale_size_continuous(limits=c(0.372,0.378))

time = read.csv("EmbBpttvsTime.csv")
time$emb.bptt = as.factor(time$emb.bptt)
ggplot(time,aes(x=epoch,y=accuracy)) + geom_line(aes(color=emb.bptt),size=1) + scale_color_manual(values=brewer.pal(n=5,"Set1")) + theme_bw()
ggplot(time,aes(x=epoch,y=accuracy)) + geom_line(aes(color=emb.bptt),size=1) + scale_color_manual(values=brewer.pal(n=5,"Set1")) + theme_bw() + coord_cartesian(xlim=c(0,9))
