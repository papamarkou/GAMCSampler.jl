type MAMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  amtune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  amfrequency::Real
end

MAMALAMCTune(smmalatune::BasicMCTune, amtune::BasicMCTune, totaltune::BasicMCTune) =
  MAMALAMCTune(smmalatune, amtune, totaltune, NaN, NaN)

MAMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  MAMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(NaN), BasicMCTune(1.), NaN, NaN)

immutable MAMALAMCTuner <: MCTuner
  smmalatuner::VanillaMCTuner
  amtuner::VanillaMCTuner
  totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
end

Base.show(io::IO, tuner::MAMALAMCTuner) = print(io, "MAMALAMCTuner")

Base.show(io::IO, ::MIME"text/plain", tuner::MAMALAMCTuner) = show(io, tuner)
