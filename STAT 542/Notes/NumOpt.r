# STAT / CES 542, Spring 2018
# This is the R code for lecture note NumOpt
# This file includes some illustrations of numerical optimizations

# look at a convex objective function for linear regression 
# the example we got from Ridge

library(MASS)
library(plot3D)
library(rgl)

set.seed(1)
n = 30
# highly correlated variables
X = mvrnorm(n, c(0, 0), matrix(c(1, 0.9999, 0.9999, 1), 2,2))
y = rnorm(n, mean= X[,1] + X[,2])


b1grid = seq(0, 2, length.out = 100)
b2grid = seq(0, 2, length.out = 100)

f = matrix(NA, length(b1grid), length(b2grid))

# OLS objective function 
for (i in 1:length(b1grid))
{
  for (j in 1:length(b2grid))
  {   
    f[i, j] = sum((y - X[,1]*b1grid[i] - X[,2]*b2grid[j])^2)
  }
}

# Ridge objective function 
lamba = 10
for (i in 1:length(b1grid))
{
  for (j in 1:length(b2grid))
  {   
    f[i, j] = sum((y - X[,1]*b1grid[i] - X[,2]*b2grid[j])^2) + lamba*b1grid[i]^2 + lamba*b2grid[j]^2
  }
}

# plot 

f = (f - min(f))/(max(f)-min(f))

M = mesh(b1grid, b2grid)
colorlut <- rainbow(102, start = 0.1)

surface3d(M$x, M$y, f*3, col = colorlut[f*100+1], alpha = 0.9, theta = 50, phi = 20, labels = c("beta 1", "beta 2", "f"))
box3d(expand = 1.1, draw_front = FALSE)

axis3d(edge = "x-+", at = seq(min(b1grid), max(b1grid), length.out = 6),
       labels = round(seq(min(b1grid), max(b1grid), length.out = 6), 2),
       tick = TRUE, line = 0, nticks = 5, cex = 1.5, adj = c(0, 0.75))

axis3d(edge = "y+-", at =  seq(min(b2grid), max(b2grid), length.out = 6),
       labels = round(seq(min(b2grid), max(b2grid), length.out = 6), 2),
       tick = TRUE, line = 0, nticks = 6, cex = 1.5, adj = c(0, -0.25))

axis3d(edge = "z+", at = 3*seq(min(f), max(f), 0.2), labels = round(seq(min(f), max(f), 0.2), 2),
       tick = TRUE, line = 1, nticks = 5, cex = 1.5, adj = 0)


mtext3d(text="beta 1", edge='y+-', line=1.5, cex = 1.5)
mtext3d(text="beta 2", edge='x-+', line=2, cex = 1.5)
mtext3d(text="Loss (scaled)", edge='z+', line=2, cex = 1.5)

segments3d(x=as.vector(c(1, 1)),
           y=as.vector(c(1, 1)),
           z=as.vector(c(0, 1)), col = "red", lwd =3)


# gradient descent vs coordinate descent 

n = 1000
p = 200
X = matrix(rnorm(n*p), n, p)
y = rnorm(n, mean=1 + X[,1] + X[,2])



GD <- function(X, y, delta, K)
{
  # the initial value of beta is all 0s
  b = matrix(0, ncol(X), 1)
  
  # I want to keep track on the objective function value at each iteration
  f = rep(NA, K)
  
  # start the iteration
  for (k in 1:K)
  {
    # the current f value 
    f[k] = sum((y - X %*% b)^2)/n/2
    
    # following our slides, this is how I update the beta in gradiant descent
    b = b + delta * t(X) %*% (y - X %*% b)
    
  }
  
  f
}

# GD(X, y, 0.0001, 1000)


CD <- function(X, y, K)
{
  # the initial value of beta is all 0s
  b = matrix(0, ncol(X), 1)
  
  # I want to keep track on the objective function value at each iteration
  f = rep(NA, K)
  
  # start the iteration
  for (k in 1:K)
  {
    # the current f value 
    f[k] = sum((y - X %*% b)^2)/n/2
    
    # following our slides, this is how I update the beta in coordinate descent
    r = y - X %*% b
    
    # this is a Gauss-Seidel style
    # if you want to use Jacob style for parallel computing, then dont need to update r within this loop.
    
    for (j in 1:ncol(X))
    {
      r = r + X[,j]*b[j]
      
      b[j] =  (X[,j] %*% r) / (X[,j]%*%X[,j])
      
      r = r - X[,j]*b[j]
    }
  }
  
  f
}

# CD(X, y, 1000)

K = 50
delta = 0.001

plot( 1:K, CD(X, y, K) , col = "blue", type = "l", lwd = 3,
        ylab = "objective function value", xlab = "iterations", cex.lab = 1.7)
		
points( 1:K, GD(X, y, delta, K) , col = "red", type = "l", lwd = 3)

legend("topright", legend = c("gradient descent", "coordinate descent"), col= c("red", "blue"), lty = 1, cex = 2)

#





