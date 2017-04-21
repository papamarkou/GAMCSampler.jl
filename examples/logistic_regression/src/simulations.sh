#!/bin/bash

JULIABIN=julia

samplers=(MALA MAMALA)

for sampler in ${samplers[@]}
do
  $JULIABIN "${HOME}/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/${sampler}/simulations.jl"
done
