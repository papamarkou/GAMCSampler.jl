ppostmean = mean(chain)

ess(chain, vtype=:bm)

ess(chain, vtype=:bm)/runtime

acceptance(chain)
