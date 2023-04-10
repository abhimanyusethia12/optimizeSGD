library(ggplot2)
library(pracma)

##################
# GENERATING DATA
##################
set.seed(1)
N <- 1e4 #sample size
d <- 1 # number of features
X <- rbind(matrix(rnorm(N*d), nrow=d, ncol=N), c(rep(1,N))) #x = (d+1)*N matrix - first value is randomly sampled from N(0,1), second value is 1 (intercept)
#theta.star = matrix(c(3,2)) #true weights 
theta.star = matrix(sample(1:10,d+1,replace = TRUE))
prob <- sigmoid(t(theta.star) %*% X + rnorm(N, sd = 2))#sigmoid(t(theta.star) %*% X + rnorm(N, sd = 2))
y <- rbinom(N,1, prob)
#glm(y ~ t(X) - 1, family = "binomial")

##################
# HELPER FUNCTIONS
##################
sigmoid <- function(z) {
  1/(1+exp(-z))
}
cost <- function(X, y, theta) {
    foo <- t(theta) %*% X
    g <- plogis(foo)
    # temp <- log(1 + exp(foo))

    (1/length(y))*sum( - y*foo + log(1 + exp(foo)))
    # (1/length(y))*sum( - y*foo - log(1 - g))
    # (1/length(y))*sum((-y*log(g)) - ((1-y)*log(1-g)))
}
cost(X,y, theta.star)
loss_grad <- function(X,y,theta){
    error <- (sigmoid(t(theta) %*% X) - y)
    delta <- X %*% t(error) / length(y)
    delta
}
loss_grad(X, y, theta.star)

##################
# HYPERPARAMETERS 
##################
theta_initial <- matrix(rep(1,d+1), nrow=d+1) #theta is initialized
batch_size <- 1#length(y)
iters_sample <- 50 #number of points plotted
iters <- as.integer(logspace(1, 4, n=iters_sample)) #equidistant points taken on log scale for plotting


##################
# VGD Function 
##################
vgd <- function(X,y,theta,batch_size,alpha,iters){
    theta_history <- list()
    theta_averaged_history <- list()
    theta <- theta_initial
    theta_sum <- 0
    theta_averaged <- theta_initial
    losses_history <- list()
    losses_averaged_history <- list()
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)]){
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]
        gradient <- loss_grad(X_batch,y_batch,theta)
        theta <- theta - alpha *gradient
        #theta_averaged <- (theta_averaged*(i-1)+theta)/i
        theta_sum <- theta_sum + theta
        theta_averaged <- theta_sum/i
        if(i %in% iters){
            theta_history[[index]] <- theta
            theta_averaged_history[[index]] <- theta_averaged
            losses_history[index] <- cost(X,y,theta)
            losses_averaged_history[index] <- cost(X,y,theta_averaged)
            index <- index +1
        }
    }

    return(list(nonaveraged = theta_history, averaged = theta_averaged_history, losses_nonaveraged = losses_history, losses_averaged = losses_averaged_history))
}

##################
# Variable LR = LR/root(K) 
##################
vgd_variablelr <- function(X,y,theta,batch_size,alpha,iters){
    theta_history <- list()
    theta <- theta_initial
    theta_averaged_history <- list()
    theta_averaged <- theta_initial
    theta_sum <- 0
    losses_history <- list()
    losses_averaged_history <- list()
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)]){
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]
        gradient <- loss_grad(X_batch,y_batch,theta)
        theta <- theta - alpha*gradient/sqrt(i)
        theta_sum <- theta_sum + theta
        theta_averaged <- theta_sum/i
        #theta_averaged <- (theta_averaged*(i-1)+theta)/i
        if(i %in% iters){
            theta_history[[index]] <- theta
            losses_history[index] <- cost(X,y,theta)
            theta_averaged_history[[index]] <- theta_averaged
            losses_averaged_history[index] <- cost(X,y,theta_averaged)
            index <- index +1
        }
        else{}
    }
    return(list(nonaveraged = theta_history, averaged = theta_averaged_history, losses_nonaveraged = losses_history, losses_averaged = losses_averaged_history))
}

##################
# Richardson Romberg Extrapolation
##################
vgd_richardsonromberg <- function(X,y,theta_initial,batch_size,alpha,iters){
    theta_history <- list()
    theta_averaged_history <- list()
    losses_history <- list()
    losses_averaged_history <- list()
    theta_gamma <- theta_initial
    theta_gamma_sum <- 0
    theta_2gamma <- theta_initial
    theta_2gamma_sum <- 0 
    theta_gamma_avg <- theta_initial
    theta_2gamma_avg <- theta_initial

    theta_rr_sum <- 0
    theta_rr_avg <- theta_initial
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)])
    {
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]

        gradient <- loss_grad(X_batch,y_batch,theta_gamma)
        theta_gamma <- theta_gamma - alpha *gradient
        theta_gamma_sum <- theta_gamma_sum + theta_gamma
        theta_gamma_avg <- theta_gamma_sum/i
        #theta_gamma_avg <- (theta_gamma_avg*i + theta_gamma)/i

        gradient <- loss_grad(X_batch,y_batch,theta_2gamma)
        theta_2gamma <- theta_2gamma - 2*alpha *gradient
        theta_2gamma_sum <- theta_2gamma_sum + theta_2gamma
        theta_2gamma_avg <- theta_2gamma_sum/i
        #theta_2gamma_avg <- (theta_2gamma_avg*i + theta_2gamma)/i

        theta_rr <- 2*theta_gamma_avg - theta_2gamma_avg
        theta_rr_sum <- theta_rr_sum + theta_rr
        theta_rr_avg <- theta_rr_sum/i
        if(i %in% iters)
        {
            theta_history[[index]] <- theta_rr
            theta_averaged_history[[index]] <- theta_rr_avg
            losses_history[index] <- cost(X,y,theta_rr)
            losses_averaged_history[index] <- cost(X,y,theta_rr_avg)
            index <- index +1
        }
    }
    return (list(nonaveraged = theta_history, averaged = theta_averaged_history,losses_nonaveraged = losses_history,losses_averaged = losses_averaged_history))
}

##################
# Richardson Romberg Extrapolation + Variable LR 
##################
vgd_vlr_richardsonromberg <- function(X,y,theta_initial,batch_size,alpha,iters){
    theta_history <- list()
    theta_averaged_history <- list()
    losses_history <- list()
    losses_averaged_history <- list()
    theta_gamma <- theta_initial
    theta_gamma_sum <- 0
    theta_2gamma <- theta_initial
    theta_2gamma_sum <- 0 
    theta_gamma_avg <- theta_initial
    theta_2gamma_avg <- theta_initial

    theta_rr_sum <- 0
    theta_rr_avg <- theta_initial
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)])
    {
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]

        gradient <- loss_grad(X_batch,y_batch,theta_gamma)
        theta_gamma <- theta_gamma - alpha *gradient/sqrt(i)
        theta_gamma_sum <- theta_gamma_sum + theta_gamma
        theta_gamma_avg <- theta_gamma_sum/i
        #theta_gamma_avg <- (theta_gamma_avg*i + theta_gamma)/i

        gradient <- loss_grad(X_batch,y_batch,theta_2gamma)
        theta_2gamma <- theta_2gamma - 2*alpha *gradient/sqrt(i)
        theta_2gamma_sum <- theta_2gamma_sum + theta_2gamma
        theta_2gamma_avg <- theta_2gamma_sum/i
        #theta_2gamma_avg <- (theta_2gamma_avg*i + theta_2gamma)/i

        theta_rr <- 2*theta_gamma_avg - theta_2gamma_avg
        theta_rr_sum <- theta_rr_sum + theta_rr
        theta_rr_avg <- theta_rr_sum/i
        if(i %in% iters)
        {
            theta_history[[index]] <- theta_rr
            theta_averaged_history[[index]] <- theta_rr_avg
            losses_history[index] <- cost(X,y,theta_rr)
            losses_averaged_history[index] <- cost(X,y,theta_rr_avg)
            index <- index +1
        }
    }
    return (list(nonaveraged = theta_history, averaged = theta_averaged_history,losses_nonaveraged = losses_history,losses_averaged = losses_averaged_history))
}

##################
# PLOTS
##################

## For alpha = 0.01
windows()
vgd_0.01 <- vgd(X,y,theta_initial,batch_size,0.01,iters)
plot(log10(iters),log10(as.numeric(vgd_0.01$losses_nonaveraged)),pch = 16, col = "red",xlab = "log_{10}k", 
    ylab = "log_{10}loss", main="Scatter Plots for Various Methods")
points(log10(iters),log10(as.numeric(vgd_0.01$losses_averaged)), pch = 16, col = "blue")

varlr_0.01 <- vgd_variablelr(X,y,theta_initial,batch_size,0.01,iters)
points(log10(iters),log10(as.numeric(varlr_0.01$losses_nonaveraged)), pch = 16, col = "green")
points(log10(iters),log10(as.numeric(varlr_0.01$losses_averaged)), pch = 16, col = "yellow")

rr_0.01 <- vgd_richardsonromberg(X,y,theta_initial,batch_size,0.01,iters)
points(log10(iters),log10(as.numeric(rr_0.01$losses_nonaveraged)), pch = 16, col = "pink")
points(log10(iters),log10(as.numeric(rr_0.01$losses_averaged)), pch = 16, col = "brown")

vlrrr_0.01 <- vgd_vlr_richardsonromberg(X,y,theta_initial,batch_size,0.01,iters)
points(log10(iters),log10(as.numeric(vlrrr_0.01$losses_nonaveraged)), pch = 16, col = "purple")
points(log10(iters),log10(as.numeric(vlrrr_0.01$losses_averaged)), pch = 16, col = "orange")

legend("bottomleft", legend = c("SGD 0.01", "SGD 0.01 Avg", "SGD 0.01/k", "SGD 0.01/k Avg","RR 0.01","RR 0.01 Avg", "RR 0.01/k","RR 0.01/k Avg"), col = c("red", "blue","green","yellow", "pink", "brown","purple","orange"), pch = 16)

## For alpha = 0.1
windows()
vgd_0.01 <- vgd(X,y,theta_initial,batch_size,0.1,iters)
plot(log10(iters),log10(as.numeric(vgd_0.01$losses_nonaveraged)),pch = 16, col = "red",xlab = "log_{10}k", 
    ylab = "log_{10}loss", main="Scatter Plots for Various Methods")
points(log10(iters),log10(as.numeric(vgd_0.01$losses_averaged)), pch = 16, col = "blue")

varlr_0.01 <- vgd_variablelr(X,y,theta_initial,batch_size,0.1,iters)
points(log10(iters),log10(as.numeric(varlr_0.01$losses_nonaveraged)), pch = 16, col = "green")
points(log10(iters),log10(as.numeric(varlr_0.01$losses_averaged)), pch = 16, col = "yellow")

rr_0.01 <- vgd_richardsonromberg(X,y,theta_initial,batch_size,0.1,iters)
points(log10(iters),log10(as.numeric(rr_0.01$losses_nonaveraged)), pch = 16, col = "pink")
points(log10(iters),log10(as.numeric(rr_0.01$losses_averaged)), pch = 16, col = "brown")

vlrrr_0.01 <- vgd_vlr_richardsonromberg(X,y,theta_initial,batch_size,0.1,iters)
points(log10(iters),log10(as.numeric(vlrrr_0.01$losses_nonaveraged)), pch = 16, col = "purple")
points(log10(iters),log10(as.numeric(vlrrr_0.01$losses_averaged)), pch = 16, col = "orange")

legend("bottomleft", legend = c("SGD 0.1", "SGD 0.1 Avg", "SGD 0.1/root(k)", "SGD 0.1/root(k) Avg","RR 0.1","RR 0.1 Avg", "RR 0.1/root(k)","RR 0.1/root(k) Avg"), col = c("red", "blue","green","yellow", "pink", "brown","purple","orange"), pch = 16)



## For best Alphas
vgd_0.3 <- vgd(X,y,theta_initial,batch_size,0.3,iters)
plot(log10(iters),log10(as.numeric(vgd_0.3$losses_nonaveraged)), pch = 16, col = "red",xlab = "log_{10}k", 
    ylab = "log_{10}loss", main="Scatter Plots for Various Methods")
points(log10(iters),log10(as.numeric(vgd_0.3$losses_averaged)), pch = 16, col = "blue")

varlr_2 <- vgd_variablelr(X,y,theta_initial,batch_size,2,iters)
points(log10(iters),log10(as.numeric(varlr_2$losses_nonaveraged)), pch = 16, col = "green")
points(log10(iters),log10(as.numeric(varlr_2$losses_averaged)), pch = 16, col = "yellow")

rr_0.6 <- vgd_richardsonromberg(X,y,theta_initial,batch_size,0.6,iters)
points(log10(iters),log10(as.numeric(rr_0.6$losses_nonaveraged)), pch = 16, col = "pink")
points(log10(iters),log10(as.numeric(rr_0.6$losses_averaged)), pch = 16, col = "brown")

vlrrr_3 <- vgd_vlr_richardsonromberg(X,y,theta_initial,batch_size,3,iters)
points(log10(iters),log10(as.numeric(vlrrr_3$losses_nonaveraged)), pch = 16, col = "purple")
points(log10(iters),log10(as.numeric(vlrrr_3$losses_averaged)), pch = 16, col = "orange")

legend("bottomleft", legend = c("SGD 0.3", "SGD 0.3 Avg", "SGD 2/root(k)", "SGD 2/root(k) Avg","RR 0.6","RR 0.6 Avg", "RR 3/root(k)","RR 3/root(k) Avg"), col = c("red", "blue","green","yellow", "pink", "brown","purple","orange"), pch = 16)
#legend("bottomleft", legend = c("SGD 0.3 Avg","SGD 2/root(k) Avg","RR 0.6 Avg","RR 3/root(k) Avg"), col = c( "blue","yellow", "brown","orange"), pch = 16)


#### Plotting theta
windows()
p <-2
plot(log10(iters),rep(theta.star[p],iters_sample),col="red",xlab = "log10(iters)", 
    ylab = "theta", main="Scatter Plots for Theta[2]",ylim = c(1,7))
points(log10(iters),sapply(vgd_0.3$averaged, function(x) x[[p]]),col='blue')
points(log10(iters),sapply(varlr_2$averaged, function(x) x[[p]]),col='green')
points(log10(iters),sapply(rr_0.6$averaged, function(x) x[[p]]),col='purple')
points(log10(iters),sapply(vlrrr_3$averaged, function(x) x[[p]]),col='orange')
legend("bottomright", legend = c("Actual", "SGD 0.3 Avg", "SGD 2/root(k) Avg","RR 0.6 Avg","RR 3/root(k) Avg"), col = c("red", "blue","green","yellow", "pink", "brown","purple","orange"), pch = 16)


windows()
p <-1
plot(log10(iters),rep(theta.star[p],iters_sample),col="red",xlab = "log10(iters)", 
    ylab = "theta", main="Scatter Plots for Theta[1]",ylim = c(1,3))
points(log10(iters),sapply(vgd_0.3$averaged, function(x) x[[p]]),col='blue')
points(log10(iters),sapply(varlr_2$averaged, function(x) x[[p]]),col='green')
points(log10(iters),sapply(rr_0.6$averaged, function(x) x[[p]]),col='purple')
points(log10(iters),sapply(vlrrr_3$averaged, function(x) x[[p]]),col='orange')
legend("topleft", legend = c("Actual", "SGD 0.3 Avg", "SGD 2/root(k) Avg","RR 0.6 Avg","RR 3/root(k) Avg"), col = c("red", "blue","green","yellow", "pink", "brown","purple","orange"), pch = 16)
