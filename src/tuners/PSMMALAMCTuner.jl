type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  malafrequency::Real
end

PSMMALAMCTune(smmalatune::BasicMCTune, malatune::BasicMCTune, totaltune::BasicMCTune) =
  PSMMALAMCTune(smmalatune, malatune, totaltune, NaN, NaN)

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.), NaN, NaN)

immutable PSMMALAMCTuner <: MCTuner
  smmalatuner::VanillaMCTuner
  malatuner::VanillaMCTuner
  totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
end

Base.show(io::IO, tuner::PSMMALAMCTuner) = print(io, "PSMMALAMCTuner")

Base.writemime(io::IO, ::MIME"text/plain", tuner::PSMMALAMCTuner) = show(io, tuner)
