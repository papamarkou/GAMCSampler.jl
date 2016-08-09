type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.))

immutable PSMMALAMCTuner <: MCTuner
  smmalatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  malatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  totaltuner::VanillaMCTuner
end

Base.show(io::IO, tuner::PSMMALAMCTuner) = print(io, "PSMMALAMCTuner")

Base.writemime(io::IO, ::MIME"text/plain", tuner::PSMMALAMCTuner) = show(io, tuner)
