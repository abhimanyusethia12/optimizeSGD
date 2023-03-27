library(ggplot2)
##################
# GENERATING DATA 
##################
N <- 10 #sample size
d <- 1 # number of features
X <- rbind(matrix(rnorm(N*d), nrow=d, ncol=N), c(rep(1,N))) #x = (d+1)*N matrix - first value is randomly sampled from N(0,1), second value is 1 (intercept)
#theta.star = matrix(c(3,2)) #true weights 
theta.star = matrix(sample(1:10,d+1,replace = TRUE))
prob <- sigmoid(t(theta.star) %*% X)
y <- rbinom(N,1, prob)
#windows()
#plot(X[1,],y, main='Plotting generated data')

##################
# COST FUNCTION
##################
sigmoid <- function(z) {
  1/(1+exp(-z))
}

cost <- function(X, y, theta) {
    g <- plogis(t(theta) %*% X)
    (1/length(y))*sum((-y*log(g)) - ((1-y)*log(1-g)))
}
cost(X,y, theta.star)
loss_grad <- function(X,y,theta){
    error <- (sigmoid(t(theta) %*% X) - y)
    delta <- X %*% t(error) / length(y)
    delta
}
convert_to_df <- function(theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper) {
    theta_history_df <- data.frame(theta0 = sapply(theta_history, `[`, 1),
                                theta1 = sapply(theta_history, `[`, 2))
    theta_history_df$loss <- apply(theta_history_df, 1, function(b) {
        cost(X, y, b)
    })

    return (theta_history_df)
}
####################
# PLOTTING FUNCTIONS
####################
xy_plot <- function(theta_history, theta_final) {
    windows()
    plot(X[1,],y, col=rgb(0.2,0.4,0.6,0.4),xlab="x[1]", ylab="y", main='Linear regression by gradient descent')
    abline(a=theta_final[2],b=theta_final[1], col='blue')
    abline(a=theta.star[2],b=theta.star[1], col='green')

    for (i in c(1,3,6,10,14,seq(20,length(theta_history),by=10))) {
        abline(a=theta_history[[i]][2],b=theta_history[[i]][1], col=rgb(0.8,0,0,0.3))
        Sys.sleep(0.1)
    }
}

##################
# HYPERPARAMETERS 
##################
num_iters <- 300 #number of iterations
alpha <- 0.01 #learning rate
#gamma <- 0.02 #momentum coefficient
theta_initial <- matrix(rep(1,d+1), nrow=d+1)
batch_size <- length(y)
#theta0_lower <- -1
#theta1_lower <- -1
#theta0_upper <- 3.5
#theta1_upper <- 2.5

##########################
# VANILLA GRADIENT DESCENT
##########################

vgd <- function(X,y,theta,batch_size, num_iters,alpha){
    theta_history <- list(num_iters)
    theta_history[[1]] <- theta
    start <- Sys.time()
    for(i in 2:num_iters){
        sample_indices <- sample(1:length(y), size = batch_size, replace = TRUE)
        X_batch <- X[,sample_indices]
        y_batch <- y[sample_indices]
        gradient <- loss_grad(X_batch,y_batch,theta)
        theta <- theta - alpha *gradient
        theta_history[[i]] <- theta
    }
    end <- Sys.time()
    return(list(theta = theta,theta_history = theta_history, time = end-start))
}
#batch_size <- 1
#num_iters <- 100
vgd_output <- vgd(X,y,theta_initial,batch_size,num_iters,alpha)
vgd_output <- vgd(X,y,theta_initial,batch_size,100,0.01)
foolog <- 10^(seq(from = 0, to = 8, length.out = 50))
foolog
cost(X,y,theta.star)
cost(X,y,vgd_output$theta)

for(k in )


vgd_output$theta
theta.star
vgd_gd <- vgd(X,y,theta_initial,length(y),num_iters,alpha)
vgd_sgd <- vgd(X,y,theta_initial,1,num_iters,alpha)


#PLOTs
xy_plot(vgd_output$theta_history,vgd_output$theta)
#contour_plotting(vgd_output$theta_history,theta0_lower, theta0_upper,theta1_lower,theta1_upper)
#sgd1 <- vgd(X,y,theta_initial,1,num_iters,alpha)
#gd1 <- vgd(X,y,theta_initial,length(y),num_iters,alpha)
#gdvssgd_contour_plotting(gd_output = gd1,sgd_output = sgd1,theta0_lower,theta0_upper,theta1_lower,theta1_upper,"Vanilla")

##################
# COMPILED RESULTS
##################
compiled_results <- data.frame(
    methods = c('Vanilla','Momentum','NAG','Adagrad','RMSprop','Adam','Corrected Adam'),
    time = c(vgd_output$time,momentum_output$time,nag_output$time,adagrad_output$time,rmsprop_output$time,adam_output$time,corrected_adam_output$time),
    loss = c(cost(X,y,vgd_output$theta),cost(X,y,momentum_output$theta),cost(X,y,nag_output$theta),cost(X,y,adagrad_output$theta),cost(X,y,rmsprop_output$theta),cost(X,y,adam_output$theta),cost(X,y,corrected_adam_output$theta))
)
print(compiled_results)
print(paste("Actual Cost: ",cost(X,y, theta.star)))

##################
# COMMON PLOT
##################
theta_grid <- expand.grid(theta0 = seq(theta0_lower, theta0_upper, length.out = 500),
                            theta1 = seq(theta1_lower, theta1_upper, length.out = 500))
theta_grid$loss <- apply(theta_grid, 1, function(b) {
    cost(X, y, b)
})
theta.star_df <- data.frame(theta0 = theta.star[1], theta1 = theta.star[2])
theta.star_df$loss <- apply(theta.star_df, 1, function(b) {
    cost(X, y, b)
})


vgd_df <- convert_to_df(vgd_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
momentum_df <- convert_to_df(momentum_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
nag_df <- convert_to_df(nag_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
adagrad_df <- convert_to_df(adagrad_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
rmsprop_df <- convert_to_df(rmsprop_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
adam_df <- convert_to_df(adam_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)
corrected_adam_df <- convert_to_df(corrected_adam_output$theta_history, theta0_lower, theta0_upper, theta1_lower, theta1_upper)

color_maps = c(
    "VGD" = "blue", 
    "Momentum" = "yellow",
    "NAG" = "orange",
    "Adagrad" = "purple",
    "RMSProp" = "pink",
    "Adam" = "brown",
    "Corrected Adam" = "maroon",
    "Actual Theta" = "green"
)
windows()
ggplot(theta_grid, aes(x = theta0, y = theta1, z = loss)) +
  geom_contour(color = "red")+
  geom_point(data = vgd_df, aes(x = theta0, y = theta1, color = "VGD"), show.legend = TRUE) +
  geom_point(data = momentum_df, aes(x = theta0, y = theta1, color = "Momentum"), show.legend = TRUE) +
  geom_point(data = nag_df,aes(x = theta0, y = theta1, color = "NAG"), show.legend = TRUE) +
  geom_point(data = adagrad_df,aes(x = theta0, y = theta1, color = "Adagrad"), show.legend = TRUE) +
  geom_point(data = rmsprop_df,aes(x = theta0, y = theta1, color = "RMSProp"), show.legend = TRUE) +
  geom_point(data = adam_df,aes(x = theta0, y = theta1, color = "Adam"), show.legend = TRUE) +
  geom_point(data = corrected_adam_df,aes(x = theta0, y = theta1, color = "Corrected Adam"), show.legend = TRUE) +
  geom_point(data = theta.star_df,aes(x = theta0, y = theta1, color = "Actual Theta"), show.legend = TRUE)+
  labs(x = expression(theta[0]),
       y = expression(theta[1]),
       title = "Contour Plot of Loss Function",
       color = "Optimization Algorithms") +
  scale_x_continuous(limits = c(theta0_lower, theta0_upper)) +
  scale_y_continuous(limits = c(theta1_lower, theta1_upper)) +
  scale_color_manual(values = color_maps)


# PLOT: Cost vs Theta[1]
plot(x=NA, y=NA, xlim=c(0,4), ylim=c(0,7), xlab="Theta [1]", ylab="Cost", main="Cost vs Theta[1] plot")
for (i in c(seq(1,num_iters,by=10))) {
    points(x=theta_history[[i]][1], y=cost_history[i], pch=20)
    Sys.sleep(0.1)
}

# PLOT: Cost vs Theta[2]
plot(x=NA, y=NA, xlim=c(0,2.2), ylim=c(0,7), xlab="Theta [2]", ylab="Cost", main="Cost vs Theta[2] plot")
for (i in c(seq(1,num_iters,by=10))) {
    points(x=theta_history[[i]][2], y=cost_history[i], pch=20)
    Sys.sleep(0.1)
}


#################
# EXPORT DATASET TO CSV
##################
dataset <- as.data.frame(X)
dataset[nrow(dataset) + 1,] <- y
write.csv(dataset, "100samples_lr_3_2.csv", row.names=FALSE)