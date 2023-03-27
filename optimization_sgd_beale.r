library(ggplot2)
##################
# HYPERPARAMETERS
##################
xx.star = matrix(c(3,0.5)) #true weights
x1_lower <- -1
x1_upper <- 3
x2_lower <- -1
x2_upper <- 1
num_iters <- 200 #number of iterations
alpha <- 0.01 #learning rate
xx_initial <- c(-1,-1)
#windows()

##################
# BEALE FUNCTION
##################
beale <- function(xx){
  # INPUT:
  # xx = c(x1, x2)
  x1 <- xx[1]
  x2 <- xx[2]	
  term1 <- (1.5 - x1 + x1*x2)^2
  term2 <- (2.25 - x1 + x1*x2^2)^2
  term3 <- (2.625 - x1 + x1*x2^3)^2
  y <- term1 + term2 + term3
  return(y)
}
beale_grad <- function(x){
    x1 <- x[1]
    x2 <- x[2]
    df1 <- 2 * (1.5 - x1 + x1 * x2) * (-1 + x2) + 2 * (2.25 - x1 + x1 * x2^2) * (-1 + x2^2) + 2 * (2.625 - x1 + x1 * x2^3) * (-1 + x2^3)
    df2 <- 2 * (1.5 - x1 + x1 * x2) * x1 + 2 * (2.25 - x1 + x1 * x2^2) * 2 * x1 * x2 + 2 * (2.625 - x1 + x1 * x2^3) * 3 * x1 * x2^2
    return(c(df1, df2))
}
####################
# PLOTTING FUNCTIONS
####################
convert_to_df <- function(xx_history, x1_lower, x1_upper, x2_lower,x2_upper) {
    xx_history_df <- data.frame(x1 = sapply(xx_history, `[`, 1),
                                x2 = sapply(xx_history, `[`, 2))
    xx_history_df$beale <- apply(xx_history_df, 1, function(b) {
        beale(b)
    })

    return (xx_history_df)
}
contour_plotting <- function(xx_history, x1_lower, x1_upper, x2_lower, x2_upper,algo_name="Optimized") {
    xx_history_df <- data.frame(x1 = sapply(xx_history, `[`, 1),
                                x2 = sapply(xx_history, `[`, 2))
    xx_history_df$beale <- apply(xx_history_df, 1, function(b) {
        beale(b)
    })  
    xx_grid <- expand.grid(x1 = seq(x1_lower, x1_upper, length.out = 500),
                            x2 = seq(x2_lower, x2_upper, length.out = 500))
    xx_grid$beale <- apply(xx_grid, 1, function(b) {
        beale(b)
    })

    xx.star_df <- data.frame(x1 = xx.star[1], x2 = xx.star[2])
    xx.star_df$beale <- apply(xx.star_df, 1, function(b) {
        beale(b)
    })

    windows()
    ggplot(xx_grid, aes(x1, x2, z = beale)) +
        geom_contour(color = "red",bins=100) +
        geom_point(data = xx_history_df,aes(x = x1, y = x2, color = algo_name), show.legend = TRUE) +
        geom_point(data=xx.star_df,aes(x = x1, y = x2,color="Optima"),show.legend = TRUE)+
        labs(x = expression(x1),
            y = expression(x2),
            title = "Contour Plot of Beale Function") +
        scale_x_continuous(limits = c(x1_lower, x1_upper)) +
        scale_y_continuous(limits = c(x2_lower, x2_upper)) +
        scale_color_manual(values = setNames(c("blue", "green"), c(algo_name, "Optima")))
        #theme_classic()
}
any2_contour_plotting <- function(xx_history1,xx_history2, x1_lower, x1_upper, x2_lower, x2_upper,name1="Method 1", name2 = "Method 2") {
    df1 <- convert_to_df(xx_history1,x1_lower,x1_upper,x2_lower,x2_upper)
    df2 <- convert_to_df(xx_history2,x1_lower,x1_upper,x2_lower,x2_upper) 
    xx_grid <- expand.grid(x1 = seq(x1_lower, x1_upper, length.out = 500),
                            x2 = seq(x2_lower, x2_upper, length.out = 500))
    xx_grid$beale <- apply(xx_grid, 1, function(b) {
        beale(b)
    })

    xx.star_df <- data.frame(x1 = xx.star[1], x2 = xx.star[2])
    xx.star_df$beale <- apply(xx.star_df, 1, function(b) {
        beale(b)
    })

    windows()
    ggplot(xx_grid, aes(x1, x2, z = beale)) +
        geom_contour(color = "red",bins=100) +
        geom_point(data = df1,aes(x = x1, y = x2, color = name1), show.legend = TRUE) +
        geom_point(data = df2,aes(x = x1, y = x2, color = name2), show.legend = TRUE) +
        geom_point(data=xx.star_df,aes(x = x1, y = x2,color="Optima"),show.legend = TRUE)+
        labs(x = expression(x1),
            y = expression(x2),
            title = "Contour Plot of Beale Function",
            color = "Legend") +
        scale_x_continuous(limits = c(x1_lower, x1_upper)) +
        scale_y_continuous(limits = c(x2_lower, x2_upper)) +
        scale_color_manual(values = setNames(c("red","blue","green"), c(name1, name2, "Optima")))
        #theme_classic()
}


##########################
# VANILLA GRADIENT DESCENT
##########################
vgd <- function(xx,num_iters,alpha){
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    start <- Sys.time()
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        xx <- xx - alpha *gradient
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}
vgd_output <- vgd(xx_initial,num_iters,alpha)
#contour_plotting(vgd_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)

##################
# MOMENTUM
##################
momentum_gd <- function(xx, num_iters, alpha,gamma){
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    start <- Sys.time()
    prev_upd <- 0
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        prev_upd <- alpha*gradient + gamma*prev_upd
        xx <- xx - prev_upd
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}
gamma <- 0.3
momentum_output <- momentum_gd(xx_initial,num_iters,alpha,gamma)
#contour_plotting(momentum_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)
#any2_contour_plotting(vgd_output$xx_history,momentum_output$xx_history,x1_lower,x1_upper,x2_lower,x2_upper,"Vanilla GD","Momentum GD")

##################
# NAG- NESTEROV ACCELERATED GRADIENT
##################
nag_gd <- function(xx, num_iters, alpha,gamma){
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    start <- Sys.time()
    prev_upd <- 0
    for(i in 2:num_iters){
        gradient <- beale_grad(xx-gamma*prev_upd)
        prev_upd <- alpha*gradient + gamma*prev_upd
        xx <- xx - prev_upd
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}
gamma <- 0.3
nag_output <- nag_gd(xx_initial,num_iters,alpha,gamma)
#contour_plotting(nag_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)

##################
# ADAGRAD
##################
adagrad <- function(xx, num_iters, alpha,epsilon) {
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    G <- c(rep(0,length(xx)))
    start <- Sys.time()
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        update <- c(rep(0,length(xx)))
        for(j in 1:length(gradient)){
            G[j] <- G[j] + (gradient[j])^2
            update[j] <- alpha*gradient[j]/sqrt(G[j]+epsilon)
        }
        xx <- xx - update
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}

alpha <- 0.3
epsilon <- 1e-8
adagrad_output <- adagrad(xx_initial,num_iters,alpha,epsilon)
#contour_plotting(adagrad_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper,"Adagrad")

##################
# RMSPROP
##################
rmsprop <- function(xx, num_iters, alpha,epsilon, gamma) {
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    G <- c(rep(0,length(xx)))
    start <- Sys.time()
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        update <- c(rep(0,length(xx)))
        for(j in 1:length(gradient)){
            G[j] <- gamma*G[j] + (1-gamma)*(gradient[j])^2
            update[j] <- alpha*gradient[j]/sqrt(G[j]+epsilon)
        }
        xx <- xx - update
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}
alpha <- 0.1
epsilon <- 1e-8
gamma <- 0.9
rmsprop_output <- rmsprop(xx_initial,num_iters,alpha,epsilon,gamma)
#contour_plotting(rmsprop_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)

##################
# ADAM
##################
adam <- function(xx, num_iters, alpha,epsilon,beta1, beta2) {
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    v <- c(rep(0,length(xx)))
    m <- c(rep(0,length(xx)))
    start <- Sys.time()
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        update <- c(rep(0,length(xx)))
        for(j in 1:length(gradient)){
            v[j] <- beta2*v[j] + (1-beta2)*(gradient[j])^2
            m[j] <- beta1*m[j] + (1-beta1)*(gradient[j])
            update[j] <- alpha*m[j]/sqrt(v[j]+epsilon)
        }
        xx <- xx - update
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}
alpha <- 0.015
epsilon <- 1e-8
beta1 <- 0.9
beta2 <- 0.999
adam_output <- adam(xx_initial,num_iters,alpha,epsilon,beta1,beta2)
#contour_plotting(adam_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)

##################
# ADAM - BIAS CORRECTED
##################
corrected_adam <- function(xx, num_iters, alpha,epsilon,beta1, beta2) {
    xx_history <- list(num_iters)
    xx_history[[1]] <- xx
    v <- c(rep(0,length(xx)))
    m <- c(rep(0,length(xx)))
    start <- Sys.time()
    for(i in 2:num_iters){
        gradient <- beale_grad(xx)
        update <- c(rep(0,length(xx)))
        for(j in 1:length(gradient)){
            v[j] <- beta2*v[j] + (1-beta2)*(gradient[j])^2
            m[j] <- beta1*m[j] + (1-beta1)*(gradient[j])
            update[j] <- alpha*(m[j]/(1-beta1))/sqrt((v[j]/(1-beta2))+epsilon)
        }
        xx <- xx - update
        xx_history[[i]] <- xx
    }
    end <- Sys.time()
    return(list(xx = xx,xx_history = xx_history, time = end-start))
}

alpha <- 0.015
epsilon <- 1e-8
beta1 <- 0.9
beta2 <- 0.999
corrected_adam_output <- corrected_adam(xx_initial,num_iters,alpha,epsilon,beta1,beta2)
contour_plotting(corrected_adam_output$xx_history,x1_lower, x1_upper, x2_lower, x2_upper)

##################
# COMPILED RESULTS
##################
compiled_results <- data.frame(
    methods = c('Vanilla','Momentum','NAG','Adagrad','RMSprop','Adam','Corrected Adam'),
    time = c(vgd_output$time,momentum_output$time,nag_output$time,adagrad_output$time,rmsprop_output$time,adam_output$time,corrected_adam_output$time),
    function_value = c(beale(vgd_output$xx),beale(momentum_output$xx),beale(nag_output$xx),beale(adagrad_output$xx),beale(rmsprop_output$xx),beale(adam_output$xx),beale(corrected_adam_output$xx))
)
print(compiled_results)
print(paste("Function Value at minima: ",beale(xx.star)))

##################
# COMMON PLOT
##################
xx_grid <- expand.grid(x1 = seq(x1_lower, x1_upper, length.out = 500),
                        x2 = seq(x2_lower, x2_upper, length.out = 500))
xx_grid$beale <- apply(xx_grid, 1, function(b) {
    beale(b)
})
xx.star_df <- data.frame(x1 = xx.star[1], x2 = xx.star[2])
xx.star_df$beale <- apply(xx.star_df, 1, function(b) {
    beale(b)
})

vgd_df <- convert_to_df(vgd_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
momentum_df <- convert_to_df(momentum_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
nag_df <- convert_to_df(nag_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
adagrad_df <- convert_to_df(adagrad_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
rmsprop_df <- convert_to_df(rmsprop_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
adam_df <- convert_to_df(adam_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)
corrected_adam_df <- convert_to_df(corrected_adam_output$xx_history, x1_lower, x1_upper, x2_lower, x2_upper)

color_maps = c(
    "VGD" = "blue", 
    "Momentum" = "yellow",
    "NAG" = "orange",
    "Adagrad" = "purple",
    "RMSProp" = "pink",
    "Adam" = "brown",
    "Corrected Adam" = "maroon",
    "Optima" = "green"
)
windows()
ggplot(xx_grid, aes(x1, x2, z = beale)) +
    geom_contour(color = "red") +
    geom_point(data = vgd_df,aes(x = x1, y = x2, color="VGD"), show.legend = TRUE) +
    geom_point(data = momentum_df,aes(x = x1, y = x2, color="Momentum"), show.legend = TRUE) +
    geom_point(data = nag_df,aes(x = x1, y = x2, color="NAG"), show.legend = TRUE) +
    geom_point(data = adagrad_df,aes(x = x1, y = x2, color="Adagrad"), show.legend = TRUE) +
    geom_point(data = rmsprop_df,aes(x = x1, y = x2, color="RMSProp"), show.legend = TRUE) +
    geom_point(data = adam_df,aes(x = x1, y = x2, color="Adam"), show.legend = TRUE) +
    geom_point(data = corrected_adam_df,aes(x = x1, y = x2, color="Corrected Adam"), show.legend = TRUE) +
    geom_point(data=xx.star_df,aes(x = x1, y = x2, color="Optima"), show.legend = TRUE)+
    labs(x = expression(x1),
        y = expression(x2),
        title = "Contour Plot of Loss Function",
        color = "Optimization Algorithms") +
    scale_x_continuous(limits = c(x1_lower, x1_upper)) +
    scale_y_continuous(limits = c(x2_lower, x2_upper)) +
    scale_color_manual(values = color_maps)
    #theme_classic()
