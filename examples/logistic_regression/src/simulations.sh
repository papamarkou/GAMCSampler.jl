#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA GAMC)

for sampler in ${samplers[@]}
do
  echo "Running $sampler..."
  $JULIABIN "$HOME/.julia/v0.6/GAMCSampler/examples/logistic_regression/src/$sampler/simulations.jl"
done
