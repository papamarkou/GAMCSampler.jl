library(data.table)
library(stringr)

DATADIR <- "data"
OUTDIR <- "output"

npars <- 4

nchains <- 10
nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 10000
ci <- 6
pi <- 2

chains <- t(fread(file.path(DATADIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE))

chainmean <- mean(chains[, pi])

submeans <- vector(mode="numeric", length=nmeans)
for (i in 1:nmeans) {
  submeans[i] <- mean(chains[1:i, pi])
}

plot(
  1:nmeans,
  submeans,
  type="l"  
)
