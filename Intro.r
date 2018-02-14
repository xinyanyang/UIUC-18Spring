# STAT/CES 542, Spring 2018
# This is the R code for lecture note Intro

# install R in your computer
# setup working directory correctly (the folder where your R code is saved)


# install necessary packages 

#install.packages("kknn") # fitting knn regression 
#install.packages("deldir") # plot Voronoi tessellation 
#install.packages("ElemStatLearn") # obtain dataset used in the HTF text book

# the "1-nearest neighbour" example: regression 

library(kknn)

# set seed
# I do this because I need to replicate the result
# You can remove this line to generate random data each time

set.seed(1)

# generate training data with 2*sin(x) and random Gaussian errors
x <- runif(15, 0, 2*pi)
y <- 2*sin(x) + rnorm(length(x))

# generate testing data points
test.x = runif(10000, 0, 2*pi)
test.y = 2*sin(test.x) + rnorm(length(test.x))

test.y = test.y[order(test.x)]
test.x = test.x[order(test.x)]

# "1-nearest neighbour" regression
k = 1
knn.fit = kknn(y ~ x, train = data.frame(x = x, y = y), test = data.frame(x = test.x, y = test.y),
				k = k, kernel = "rectangular")
test.pred = knn.fit$fitted.values

# plot the data
par(mar = rep(2,4))
plot(x, y, xlim = c(0, 2*pi), pch = "o", cex = 2, xlab = "", ylab = "", cex.lab = 1.5)
title(main = paste(k, "-Nearest Neighbor Regression", sep = ""), cex.main = 1.5)

# plot the true regression line
lines(test.x, 2*sin(test.x), col = "deepskyblue", lwd = 3)
box()

# plot the fitted line
lines(test.x, test.pred, type = "s", col = "darkorange", lwd = 3)



# now we generate more data
set.seed(1)
x <- runif(200, 0, 2*pi)
y <- 2*sin(x) + rnorm(length(x))
test.x = runif(10000, 0, 2*pi)
test.y = 2*sin(test.x) + rnorm(length(test.x))

# I reorder the testing data according to the value of X, so that I can plot them easier later on
test.y = test.y[order(test.x)]
test.x = test.x[order(test.x)]

# 1-nearest neighbour --- you can try different k values and see the results
k = 1
knn.fit = kknn(y ~ x, train = data.frame(x = x, y = y), test = data.frame(x = test.x, y = test.y),
				k = k, kernel = "rectangular")
test.pred = knn.fit$fitted.values
par(mar=rep(2,4))
plot(x, y, xlim = c(0, 2*pi), pch = 19, cex = 1, axes=FALSE, ylim = c(-4.25, 4.25))
title(main=paste(k, "-Nearest Neighbor Regression", sep = ""))
lines(test.x, 2*sin(test.x), col = "deepskyblue", lwd = 3)
lines(test.x, test.pred, type = "s", col = "darkorange", lwd = 3)
box()

# prediction error
mean((test.pred - test.y)^2)


######################## Cross Validation ###########################

# 10 fold cross validation

nfold = 10
infold = sample(rep(1:nfold, length.out=length(x)))

mydata = data.frame(x = x, y = y)

K = 50 # maximum number of k that I am considering
errorMatrix = matrix(NA, K, nfold) # save the prediction error of each fold

for (l in 1:nfold)
{
	for (k in 1:K)
	{
		knn.fit = kknn(y ~ x, train = mydata[infold != l, ], test = mydata[infold == l, ], k = k)
		errorMatrix[k, l] = mean((knn.fit$fitted.values - mydata$y[infold == l])^2)
	}
}

# plot the results
plot(rep(1:K, nfold), as.vector(errorMatrix), pch = 19, cex = 0.5)
points(1:K, apply(errorMatrix, 1, mean), col = "red", pch = 19, type = "l", lwd = 3)

# which k is the best?
which.min(apply(errorMatrix, 1, mean))


# bootstrapped cross validation
nsim = 100
errorMatrix = matrix(NA, K, nsim) # save the prediction error of each fold

for (l in 1:nsim)
{
	testid = sample(1:nrow(mydata), 0.1*nrow(mydata))
	for (k in 1:K)
	{
		knn.fit = kknn(y ~ x, train = mydata[-testid, ], test = mydata[testid, ], k = k)
		errorMatrix[k, l] = mean((knn.fit$fitted.values - mydata$y[testid])^2)
	}
}

# plot the results
plot(rep(1:K, nsim), as.vector(errorMatrix), pch = 19, cex = 0.5)
points(1:K, apply(errorMatrix, 1, mean), col = "red", pch = 19, type = "l", lwd = 3)

# which k is the best?
which.min(apply(errorMatrix, 1, mean))







###################################################
# knn for classification: 

# the "1-nearest neighbour" example

library(class)
set.seed(1)

# generate 20 random observations, with random class 1/0
x <- matrix(runif(40), 20, 2)
g <- rbinom(20, 1, 0.5)

# generate a grid for plot
xgd1 = xgd2 = seq(0, 1, 0.01)
gd = expand.grid(xgd1, xgd2)

# fit a 1-nearest neighbour model and get the fitted class
knn1 <- knn(x, gd, g, k=1)
knn1.class <- matrix(knn1, length(xgd1), length(xgd2))

# plot the data 
plot(x, col=ifelse(g==1, "darkorange", "deepskyblue"), pch = 19, cex = 3, axes = FALSE, xlim= c(0, 1), ylim = c(0, 1))
symbols(0.7, 0.7, circles = 2, add = TRUE)
points(0.7, 0.7, pch = 19)
box()

# Voronoi tessalation plot (1NN)
library(deldir)
par(mar=rep(2,4))
z <- deldir(x = data.frame(x = x[,1], y = x[,2], z=as.factor(g)), rw = c(0, 1, 0, 1))
w <- tile.list(z)
plot(w, fillcol=ifelse(g==1, "bisque", "cadetblue1"), axes=FALSE, labels = "")
points(x, col=ifelse(g==1, "darkorange", "deepskyblue"), pch = 19, cex = 3)




# Example from HTF text book
library(ElemStatLearn)
library(class)

x <- mixture.example$x
y <- mixture.example$y
xnew <- mixture.example$xnew

par(mar=rep(2,4))
plot(x, col=ifelse(y==1, "darkorange", "deepskyblue"), axes = FALSE)
box()

# knn classification 

k = 15
knn.fit <- knn(x, xnew, y, k=k)

px1 <- mixture.example$px1
px2 <- mixture.example$px2
pred <- matrix(knn.fit == "1", length(px1), length(px2))

contour(px1, px2, pred, levels=0.5, labels="",axes=FALSE)
box()
title(paste(k, "-Nearest Neighbour", sep= ""))
points(x, col=ifelse(y==1, "darkorange", "deepskyblue"))
mesh <- expand.grid(px1, px2)
points(mesh, pch=".", cex=1.2, col=ifelse(pred, "darkorange", "deepskyblue"))


# using linear regression to fit the data (not logistic)

lm.fit = glm(y~x)
lm.pred = matrix(as.matrix(cbind(1, xnew)) %*% as.matrix(lm.fit$coef) > 0.5, length(px1), length(px2))

par(mar=rep(2,4))
plot(mesh, pch=".", cex=1.2, col=ifelse(lm.pred, "darkorange", "deepskyblue"), axes=FALSE)

abline(a = (0.5 - lm.fit$coef[1])/lm.fit$coef[3], b = -lm.fit$coef[2]/lm.fit$coef[3], lwd = 2)
points(x, col=ifelse(y==1, "darkorange", "deepskyblue"))
title("Linear Regression of 0/1 Response")
box()







# Handwritten Digit Recognition Data
library(ElemStatLearn)
# the first column is the true digit
dim(zip.train)
dim(zip.test)

# look at one sample

image(zip2image(zip.train, 1), col=gray(256:0/256), zlim=c(0,1), xlab="", ylab="", axes = FALSE)


# a plot of some samples 
findRows <- function(zip, n) {
# Find n (random) rows with zip representing 0,1,2,...,9
res <- vector(length=10, mode="list")
names(res) <- 0:9
ind <- zip[,1]
for (j in 0:9) {
res[[j+1]] <- sample( which(ind==j), n ) }
return(res) }

# Making a plot like that on page 4 of HTF:
digits <- vector(length=10, mode="list")
names(digits) <- 0:9
rows <- findRows(zip.train, 6)
for (j in 0:9) {
digits[[j+1]] <- do.call("cbind", lapply(as.list(rows[[j+1]]),
function(x) zip2image(zip.train, x)) )
}
im <- do.call("rbind", digits)

image(im, col=gray(256:0/256), zlim=c(0,1), xlab="", ylab="" )


# fit knn model and calculate error

k = 3

knn.fit <- knn(zip.train[, 2:257], zip.test[, 2:257], zip.train[, 1], k=k)

mean(knn.fit != zip.test[,1])

table(knn.fit, zip.test[,1])

