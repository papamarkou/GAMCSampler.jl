#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA GAMC)

for sampler in ${samplers[@]}
do
  echo "Generating $sampler table line..."
  $JULIABIN "$HOME/.julia/v0.6/GAMCSampler/examples/logistic_regression/src/$sampler/table.jl"
done
