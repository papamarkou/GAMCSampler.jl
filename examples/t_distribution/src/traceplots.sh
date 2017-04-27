#!/bin/bash

RBIN=Rscript

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Gennerating $sampler traceplot..."
  $RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/t_distribution/src/$sampler/traceplot.r"
done
