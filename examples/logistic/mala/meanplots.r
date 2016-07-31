library(data.table)

x <- t(fread("data/chain10.csv", sep=",", header=FALSE))
mean(x[, 2])
