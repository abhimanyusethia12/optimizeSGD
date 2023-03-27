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
prob <- sigmoid(t(theta.star) %*% X + rnorm(N, sd = 2))
y <- rbinom(N,1, prob)
 glm(y ~ t(X) - 1, family = "binomial")

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
theta_initial <- matrix(rep(1,d+1), nrow=d+1)
batch_size <- 1#length(y)
iters_sample <- 50
iters <- as.integer(logspace(1, 4, n=iters_sample))


vgd <- function(X,y,theta,batch_size,alpha,iters){
    theta_history <- list()
    theta_averaged_history <- list()
    theta <- theta_initial
    theta_averaged <- theta_initial
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)]){
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]
        gradient <- loss_grad(X_batch,y_batch,theta)
        theta <- theta - alpha *gradient
        theta_averaged <- (theta_averaged*(i-1)+theta)/i
        if(i %in% iters){
            theta_history[[index]] <- theta
            theta_averaged_history[[index]] <- theta_averaged
            index <- index +1
        }
    }
    return(list(nonaveraged = theta_history, averaged = theta_averaged_history))
}

vgd_output <- vgd(X,y,theta_initial,batch_size,0.01,iters)
theta_0.01_nonavg <- vgd_output$nonaveraged
theta_0.01_avg <- vgd_output$averaged
vgd_output <- vgd(X,y,theta_initial,batch_size,0.1,iters)
theta_0.1_nonavg <- vgd_output$nonaveraged
theta_0.1_avg <- vgd_output$averaged

vgd_variablelr <- function(X,y,theta,batch_size,alpha,iters){
    theta_history <- list()
    theta <- theta_initial
    theta_averaged_history <- list()
    theta_averaged <- theta_initial

    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)]){
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]
        gradient <- loss_grad(X_batch,y_batch,theta)
        theta <- theta - alpha *gradient/sqrt(i)
        theta_averaged <- (theta_averaged*(i-1)+theta)/i
        if(i %in% iters){
            theta_history[[index]] <- theta
            theta_averaged_history[[index]] <- theta_averaged
            index <- index +1
        }
        else{}
    }
    return(list(nonaveraged = theta_history, averaged = theta_averaged_history))
}
theta_0.1varlr_output <- vgd(X,y,theta_initial,batch_size,0.1,iters)
theta_0.1varlr_nonavg <- theta_0.1varlr_output$nonaveraged
theta_0.1varlr_avg <- theta_0.1varlr_output$averaged

vgd_richardsonromberg <- function(X,y,theta_initial,batch_size,alpha,iters){
    theta_history <- list()
    theta_averaged_history <- list()
    theta_gamma <- theta_initial
    theta_2gamma <- theta_initial
    theta_gamma_avg <- theta_initial
    theta_2gamma_avg <- theta_initial
    index <- 1
    #start <- Sys.time()
    for(i in 1:iters[length(iters)])
    {
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]

        gradient <- loss_grad(X_batch,y_batch,theta_gamma)
        theta_gamma <- theta_gamma - alpha *gradient
        theta_gamma_avg <- (theta_gamma_avg*i + theta_gamma)/i

        gradient <- loss_grad(X_batch,y_batch,theta_2gamma)
        theta_2gamma <- theta_2gamma - 2*alpha *gradient
        theta_2gamma_avg <- (theta_2gamma_avg*i + theta_2gamma)/i

        if(i %in% iters)
        {
            theta_history[[index]] <- 2*theta_gamma - theta_2gamma
            theta_averaged_history[[index]] <- 2*theta_gamma_avg - theta_2gamma_avg
            index <- index +1
        }
    }
    return (list(nonaveraged = theta_history, averaged = theta_averaged_history))
}

theta_rr_output <- vgd_richardsonromberg(X,y,theta_initial,batch_size,0.1,iters)
theta_0.1rr_nonavg <- theta_rr_output$nonaveraged
theta_0.1rr_avg <- theta_rr_output$averaged


##PLOTTING
losses_0.01 <- list()
losses_0.1 <- list()
losses_0.1varlr <- list()
losses_0.01_avg <- list()
losses_0.1_avg <- list()
losses_0.1varlr_avg <- list()
losses_0.1rr <- list()
losses_0.1rr_avg <- list()
for(i in 1:iters_sample){
    losses_0.01[i] <- cost(X,y,theta_0.01_nonavg[[i]])
    losses_0.1[i] <- cost(X,y,theta_0.1_nonavg[[i]]) 
    losses_0.1varlr[i] <- cost(X,y,theta_0.1varlr_nonavg[[i]])
    losses_0.01_avg[i] <- cost(X,y,theta_0.01_avg[[i]])
    losses_0.1_avg[i] <- cost(X,y,theta_0.1_avg[[i]]) 
    losses_0.1varlr_avg[i] <- cost(X,y,theta_0.1varlr_avg[[i]])
    losses_0.1rr[i] <- cost(X,y,theta_0.1rr_nonavg[[i]])
    losses_0.1rr_avg[i] <- cost(X,y,theta_0.1rr_avg[[i]])
}

windows()
plot(log10(iters),log10(as.numeric(losses_0.01)),pch = 16, col = "red",xlab = "log_{10}k", 
    ylab = "log_{10}loss", main="Scatter Plots for Various LR")
points(log10(iters),log10(as.numeric(losses_0.01_avg)), pch = 16, col = "blue")
#plot(log10(iters),log10(as.numeric(losses_0.1)),pch = 16, col = "red",xlab = "log_{10}k", ylab = "log_{10}loss", main="Scatter Plots for Various LR")
points(log10(iters),log10(as.numeric(losses_0.1)), pch = 16, col = "green")
points(log10(iters),log10(as.numeric(losses_0.1_avg)), pch = 16, col = "yellow")
points(log10(iters),log10(as.numeric(losses_0.1varlr)), pch = 16, col = "purple")
points(log10(iters),log10(as.numeric(losses_0.1varlr_avg)), pch = 16, col = "orange")
points(log10(iters),log10(as.numeric(losses_0.1rr)), pch = 16, col = "pink")
points(log10(iters),log10(as.numeric(losses_0.1rr_avg)), pch = 16, col = "brown")

#legend("topright", legend = c("Alpha = 0.01", "Alpha = 0.1", "Alpha = 0.1/k","Alpha = 0.01 Avg", "Alpha = 0.1 Avg", "Alpha = 0.1/k Avg"), col = c("blue", "red","yellow","violet","maroon","green"), pch = 16)
legend("bottomleft", legend = c("Alpha = 0.01", "Alpha = 0.01 Avg", "Alpha = 0.1", "Alpha = 0.1 Avg","Alpha = 0.1/k","Alpha = 0.1/k Avg", "Richardson with Alpha = 0.1","Richardson Averaged with Alpha = 0.1"), col = c("blue", "red","green","yellow", "purple", "orange","pink","brown"), pch = 16)
#legend("bottomleft", legend = c("Alpha = 0.1", "Alpha = 0.1 Avg", "Richardson with Alpha = 0.1","Richardson Averaged with Alpha = 0.1"), col = c( "red","maroon", "yellow", "green"), pch = 16)
