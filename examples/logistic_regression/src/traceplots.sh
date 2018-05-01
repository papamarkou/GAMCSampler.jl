#!/bin/bash

RBIN=Rscript

samplers=(AM MALA SMMALA GAMC)

for sampler in ${samplers[@]}
do
  echo "Gennerating $sampler traceplot..."
  $RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/logistic_regression/src/$sampler/traceplot.r"
done
