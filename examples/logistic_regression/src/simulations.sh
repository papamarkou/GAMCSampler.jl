#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Running $sampler..."
  $JULIABIN "$HOME/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/$sampler/simulations.jl"
done
