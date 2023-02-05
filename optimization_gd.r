##################
# GENERATING DATA 
##################
N <- 1000 #sample size
X <- rbind(matrix(rnorm(N,mean=0, sd=1), nrow=1, ncol=N), c(rep(1,N))) #x = N*2 matrix - first value is randomly sampled from N(0,1), second value is 1 (intercept)
theta.star = matrix(c(3,2)) #true weights 
y <- rnorm(N, mean= t(theta.star) %*% X, sd=1) #generated output with gaussian noise

plot(X[1,],y, main='Plotting generated data')


##################
# COST FUNCTION
##################
# squared error cost function
cost <- function(X, y, theta) {
  sum( (t(theta) %*% X - y)^2 ) / (2*length(y))
}
#cost <- function(X,y, theta){
#    sum((theta[1]*X + theta[2] - y)^2)/(2*length(y))
#}
cost(X,y, theta.star)

##################
# HYPERPARAMETERS 
##################
num_iters <- 500 #number of iterations
alpha <- 0.01 #learning rate
gamma <- 0.02 #momentum coefficient

##################
# VANILLA GRADIENT DESCENT
##################
vgd_update <- function(X, y, theta, alpha) {
    error <- ( t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    alpha* delta
}  

theta <- matrix(c(0,0), nrow=2) #initializing
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
for (i in 1:num_iters) {
    theta <- theta - vgd_update(X,y,theta,alpha)
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
}
end <- Sys.time()

vgd_time <- end - start
vgd_cost <- cost(X,y,theta)
print(vgd_time)
print(theta)
print(vgd_cost)

##################
# MOMENTUM
##################
momentum_update <- function(X, y, theta, alpha, gamma, prev_upd) {
    error <- ( t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    alpha*delta + gamma*prev_upd
}  

theta <- matrix(c(0,0), nrow=2) #initializing
start <- Sys.time()
prev_upd <- 0
for (i in 1:num_iters) {
    prev_upd <- momentum_update(X,y,theta,alpha, gamma, prev_upd)
    theta <- theta - prev_upd
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
}
end <- Sys.time()

momentum_time <- end - start
momentum_cost <- cost(X,y,theta)
print(momentum_time)
print(theta)
print(momentum_cost)

##################
# NAG- NESTEROV ACCELERATED GRADIENT
##################
nag_update <- function(X, y, theta, alpha, gamma, prev_upd) {
    error <- ( t(theta- gamma*prev_upd) %*% X - y)
    delta <- X %*% t(error) / length(y)
    alpha*delta + gamma*prev_upd
}  

theta <- matrix(c(0,0), nrow=2) #initializing
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
prev_upd <- 0
for (i in 1:num_iters) {
    prev_upd <- nag_update(X,y,theta,alpha, gamma, prev_upd)
    theta <- theta - prev_upd
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
}
end <- Sys.time()

nag_time <- end - start
nag_cost <- cost(X,y,theta)
print(nag_time)
print(theta)
print(nag_cost)

##################
# ADAGRAD
##################

theta <- matrix(c(0,0), nrow=2) #initializing
G <- c(rep(0,dim(X)[1]))
epsilon <- 1
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
for (i in 1:num_iters) {
    error <- (t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    update <- matrix(c(rep(0,dim(X)[1])),nrow=dim(X)[1])
    for(j in 1:length(delta)){
        update[j] <- alpha*delta[j]/(sqrt(G[j]+epsilon))
        G[j] <- G[j] + (delta[j])^2
    }
    theta <- theta - update
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
}
end <- Sys.time()

adagrad_time <- end - start
adagrad_cost <- cost(X,y,theta)
print(adagrad_time)
print(theta)
print(adagrad_cost)


##################
# RMSPROP
##################

theta <- matrix(c(0,0), nrow=2) #initializing
G <- c(rep(0,dim(X)[1]))
epsilon <- 1
gamma <- 0.9
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
for (i in 1:num_iters) {
    error <- (t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    update <- matrix(c(rep(0,dim(X)[1])),nrow=dim(X)[1])
    for(j in 1:length(delta)){
        update[j] <- alpha*delta[j]/(sqrt(G[j]+epsilon))
        G[j] <- gamma*G[j] + (1-gamma)*(delta[j])^2
    }
    theta <- theta - update
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
    #print(G)
}
end <- Sys.time()

rmsprop_time <- end - start
rmsprop_cost <- cost(X,y,theta)
print(rmsprop_time)
print(theta)
print(rmsprop_cost)

##################
# ADAM
##################

theta <- matrix(c(0,0), nrow=2) #initializing
v <- c(rep(0,dim(X)[1]))
m <- c(rep(0,dim(X)[1]))
epsilon <- 1
beta1 <- 0.9
beta2 <- 0.999
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
for (i in 1:num_iters) {
    error <- (t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    update <- matrix(c(rep(0,dim(X)[1])),nrow=dim(X)[1])
    for(j in 1:length(delta)){
        update[j] <- alpha*m[j]/(sqrt(v[j]+epsilon))
        v[j] <- beta2*v[j] + (1-beta2)*(delta[j])^2
        m[j] <- beta1*m[j] + (1-beta1)*(delta[j])
    }
    theta <- theta - update
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
    #print(G)
}
end <- Sys.time()

adam_time <- end - start
adam_cost <- cost(X,y,theta)
print(adam_time)
print(theta)
print(adam_cost)

##################
# ADAM - BIAS CORRECTED
##################

theta <- matrix(c(0,0), nrow=2) #initializing
v <- c(rep(0,dim(X)[1]))
m <- c(rep(0,dim(X)[1]))
epsilon <- 1
beta1 <- 0.9
beta2 <- 0.999
cost_history <- double(num_iters)
theta_history <- list(num_iters)

start <- Sys.time()
for (i in 1:num_iters) {
    error <- (t(theta) %*% X - y)
    delta <- X %*% t(error) / length(y)
    update <- matrix(c(rep(0,dim(X)[1])),nrow=dim(X)[1])
    for(j in 1:length(delta)){
        update[j] <- alpha*(m[j]/1-beta1)/(sqrt((v[j]/1-beta2)+epsilon))
        v[j] <- beta2*v[j] + (1-beta2)*(delta[j])^2
        m[j] <- beta1*m[j] + (1-beta1)*(delta[j])
    }
    theta <- theta - update
    cost_history[i] <- cost(X, y, theta)
    theta_history[[i]] <- theta
    #print(G)
}
end <- Sys.time()

biascorradam_time <- end - start
biascorradam_cost <- cost(X,y,theta)
print(biascorradam_time)
print(theta)
print(biascorradam_cost)


##################
# COMPILED RESULTS
##################
print(paste("Vanilla GD Time: ",vgd_time))
print(paste("Momentum GD Time: ",momentum_time))
print(paste("NAG GD Time: ",nag_time))
print(paste("Adagrad GD Time: ",adagrad_time))
print(paste("RMSProp GD Time: ",rmsprop_time))
print(paste("Adam GD Time: ",adam_time))
print(paste("Bias-Corrected Adam GD Time: ",biascorradam_time))

print(paste("Actual Cost: ",cost(X,y, theta.star)))
print(paste("Vanilla GD Cost: ",vgd_cost))
print(paste("Momentum GD Cost: ",momentum_cost))
print(paste("NAG GD Cost: ",nag_cost))
print(paste("Adagrad GD Cost: ",adagrad_cost))
print(paste("RMSProp GD Cost: ",rmsprop_cost))
print(paste("Adam GD Cost: ",adam_cost))
print(paste("Bias-Corrected Adam GD Cost: ",biascorradam_cost))

##################
# PLOTS
##################
plot(X[1,],y, col=rgb(0.2,0.4,0.6,0.4),xlab="theta[1]", ylab="y", main='Linear regression by gradient descent')
abline(coef=theta, col='blue')
abline(coef=theta.star, col='green')

for (i in c(1,3,6,10,14,seq(20,num_iters,by=10))) {
  abline(coef=theta_history[[i]], col=rgb(0.8,0,0,0.3))
  Sys.sleep(0.1)
}

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
