function codegen(::Type{Val{:iterate}}, ::Type{GAMC}, job::BasicMCJob)
  local result::Expr
  burninbody = []
  ambody = []
  smmalapasttensorbody = []
  smmalabody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in GAMC code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    push!(body, :(_job.sstate.tune.totaltune.proposed += 1))
  end

  push!(body, :(
    if _job.sstate.count > _job.sampler.t0
      _job.sampler.update!(_job.sstate, _job.pstate, _job.sstate.count, _job.range.nsteps)
    end
  ))

  push!(smmalabody, :(_job.sstate.updatetensorcount += 1))

  if job.tuner.smmalatuner.verbose
    push!(smmalabody, :(_job.sstate.tune.smmalatune.proposed += 1))
  end

  push!(smmalapasttensorbody, :(_job.parameter.uptotensorlogtarget!(_job.pstate)))

  if job.sampler.transform != nothing
    push!(smmalapasttensorbody, :(_job.pstate.tensorlogtarget[:, :] = _job.sampler.transform(_job.pstate.tensorlogtarget)))
  end

  push!(smmalapasttensorbody, :(_job.sstate.oldinvtensor[:, :] = inv(_job.pstate.tensorlogtarget)))

  push!(smmalapasttensorbody, :(_job.sstate.oldfirstterm[:] = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget))

  push!(smmalapasttensorbody, :(_job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(_job.sstate.oldinvtensor)))))

  push!(smmalabody, Expr(:if, :(!_job.sstate.pastupdatetensor), Expr(:block, smmalapasttensorbody...)))

  push!(smmalabody, :(_job.sstate.μ[:] = _job.pstate.value+0.5*_job.sstate.tune.totaltune.step*_job.sstate.oldfirstterm))

  push!(
    smmalabody,
    :(_job.sstate.pstate.value[:] = _job.sstate.μ+_job.sstate.sqrttunestep*_job.sstate.cholinvtensor*randn(_job.pstate.size))
  )

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.sstate.pstate)))

  if job.sampler.transform != nothing
    push!(
      smmalabody,
      :(_job.sstate.pstate.tensorlogtarget[:, :] = _job.sampler.transform(_job.sstate.pstate.tensorlogtarget))
    )
  end

  push!(smmalabody, :(_job.sstate.newinvtensor[:, :] = inv(_job.sstate.pstate.tensorlogtarget)))

  push!(smmalabody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    smmalabody,
    :(
      _job.sstate.ratio += (
        0.5*(
          logdet(_job.sstate.tune.totaltune.step*_job.sstate.oldinvtensor)
          +dot(
            _job.sstate.pstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.sstate.pstate.value-_job.sstate.μ)
          )/_job.sstate.tune.totaltune.step
        )
      )
    )
  )

  push!(smmalabody, :(_job.sstate.newfirstterm[:] = _job.sstate.newinvtensor*_job.sstate.pstate.gradlogtarget))

  push!(
    smmalabody,
    :(_job.sstate.μ[:] = _job.sstate.pstate.value+0.5*_job.sstate.tune.totaltune.step*_job.sstate.newfirstterm)
  )

  push!(smmalabody, :(
      _job.sstate.ratio -= (
        0.5*(
          logdet(_job.sstate.tune.totaltune.step*_job.sstate.newinvtensor)
          +dot(
            _job.pstate.value-_job.sstate.μ,
            _job.sstate.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
          )/_job.sstate.tune.totaltune.step
        )
      )
    )
  )

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))

  if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
  end

  if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
  end

  push!(update, :(_job.pstate.tensorlogtarget = copy(_job.sstate.pstate.tensorlogtarget)))

  if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
    push!(update, :(_job.pstate.tensorloglikelihood = copy(_job.sstate.pstate.tensorloglikelihood)))
  end

  if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
    push!(update, :(_job.pstate.tensorlogprior = copy(_job.sstate.pstate.tensorlogprior)))
  end

  push!(update, :(_job.sstate.oldinvtensor = copy(_job.sstate.newinvtensor)))

  push!(update, :(_job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(_job.sstate.newinvtensor)))))

  push!(update, :(_job.sstate.oldfirstterm = copy(_job.sstate.newfirstterm)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  dindex = findfirst(job.outopts[:diagnostics], :accept)
  if dindex != 0
    push!(update, :(_job.pstate.diagnosticvalues[$dindex] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[$dindex] = false))
  end

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.smmalatuner.verbose
    push!(update, :(_job.sstate.tune.smmalatune.accepted += 1))
  end

  push!(
    smmalabody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  if job.tuner.amtuner.verbose
    push!(ambody, :(_job.sstate.tune.amtune.proposed += 1))
  end

  push!(
    ambody,
    :(
      covariance!(
        _job.sstate.oldinvtensor,
        _job.sstate.oldinvtensor,
        _job.sstate.count-2,
        _job.pstate.value,
        _job.sstate.lastmean,
        _job.sstate.secondlastmean
      )
    )
  )

  push!(ambody, :(_job.sstate.oldinvtensor[:, :] = Hermitian(_job.sstate.oldinvtensor)))

  push!(ambody, :(set_gmm!(_job.sstate, _job.sampler, _job.pstate)))

  push!(ambody, :(_job.sstate.pstate.value[:] =  rand(_job.sstate.proposal)))

  push!(ambody, :(_job.parameter.logtarget!(_job.sstate.pstate)))

  push!(ambody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(ambody, :(_job.sstate.ratio -= logpdf(_job.sstate.proposal, _job.sstate.pstate.value)))

  push!(ambody, :(set_gmm!(_job.sstate, _job.sampler, _job.sstate.pstate)))

  push!(ambody, :(_job.sstate.ratio += logpdf(_job.sstate.proposal, _job.pstate.value)))

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  dindex = findfirst(job.outopts[:diagnostics], :accept)
  if dindex != 0
    push!(update, :(_job.pstate.diagnosticvalues[$dindex] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[$dindex] = false))
  end

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.amtuner.verbose
    push!(update, :(_job.sstate.tune.amtune.accepted += 1))
  end

  push!(
    ambody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  push!(body, Expr(:if, :(_job.sstate.presentupdatetensor), Expr(:block, smmalabody...), Expr(:block, ambody...)))

  push!(body, :(_job.sstate.secondlastmean = copy(_job.sstate.lastmean)))

  push!(body, :(recursive_mean!(_job.sstate.lastmean, _job.sstate.lastmean, _job.sstate.count, _job.pstate.value)))

  push!(body, :(_job.sstate.pastupdatetensor = _job.sstate.presentupdatetensor))

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune.totaltune)))
  end

  if job.tuner.smmalatuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.smmalafrequency = _job.sstate.tune.smmalatune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.amtuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.amfrequency = _job.sstate.tune.amtune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.amtune)))
  end

  if job.tuner.totaltuner.verbose || job.tuner.smmalatuner.verbose || job.tuner.amtuner.verbose
    fmt_tot_iter = format_iteration(ndigits(job.range.burnin))
    fmt_burnin_iter = format_iteration(ndigits(job.tuner.totaltuner.period))
    fmt_perc = format_percentage()

    push!(burninbody, :(println(
      "Burnin iteration ",
      $(fmt_tot_iter)(_job.sstate.count),
      " out of ",
      _job.range.burnin,
      "..."
    )))

    if job.tuner.totaltuner.verbose
      push!(burninbody, :(println(
        "  Total : ",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.totaltune.rate),
        "%) acceptance rate"
      )))
    end

    if job.tuner.smmalatuner.verbose
      push!(burninbody, :(println(
        "  SMMALA: ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalatune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalafrequency),
        "%) running frequency"
      )))
    end

    if job.tuner.amtuner.verbose
      push!(burninbody, :(println(
        "  AM    : ",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.amtune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.amfrequency),
        "%) running frequency"
      )))
    end
  end

  if isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune.totaltune, _job.tuner.totaltuner)))
    push!(burninbody, :(_job.sstate.sqrttunestep = sqrt(_job.sstate.tune.totaltune.step)))
  end

  if job.tuner.smmalatuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.amtuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.amtune)))
  end

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    push!(burninbody, :(_job.sstate.tune.totaltune.totproposed += _job.sstate.tune.totaltune.proposed))

    push!(burninbody, :(_job.sstate.tune.totaltune.proposed = 0))

    if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
      push!(burninbody, :(_job.sstate.tune.totaltune.accepted = 0))
      push!(burninbody, :(_job.sstate.tune.totaltune.rate = NaN))
    end

    push!(
      body,
      Expr(
        :if,
        :(
          _job.sstate.tune.totaltune.totproposed <= _job.range.burnin &&
          mod(_job.sstate.tune.totaltune.proposed, _job.tuner.totaltuner.period) == 0
        ),
        Expr(:block, burninbody...)
      )
    )
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end

function iterate!(job::BasicMCJob, ::Type{GAMC}, ::Type{Multivariate})
  job.sstate.count += 1

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    job.sstate.tune.totaltune.proposed += 1
  end

  if job.sstate.count > job.sampler.t0
    job.sampler.update!(job.sstate, job.pstate, job.sstate.count, job.range.nsteps)
  end

  if job.sstate.presentupdatetensor
    # smmalabody
    job.sstate.updatetensorcount += 1

    if job.tuner.smmalatuner.verbose
      job.sstate.tune.smmalatune.proposed += 1
    end

    if !_job.sstate.pastupdatetensor
      # smmalapasttensorbody
      job.parameter.uptotensorlogtarget!(job.pstate)

      if job.sampler.transform != nothing
        job.pstate.tensorlogtarget[:, :] = job.sampler.transform(job.pstate.tensorlogtarget)
      end

      job.sstate.oldinvtensor[:, :] = inv(job.pstate.tensorlogtarget)

      job.sstate.oldfirstterm[:] = job.sstate.oldinvtensor*job.pstate.gradlogtarget

      job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(job.sstate.oldinvtensor)))
    end

    job.sstate.μ[:] = job.pstate.value+0.5*job.sstate.tune.totaltune.step*job.sstate.oldfirstterm

    job.sstate.pstate.value[:] = job.sstate.μ+job.sstate.sqrttunestep*job.sstate.cholinvtensor*randn(job.pstate.size)

    job.parameter.uptotensorlogtarget!(job.sstate.pstate)

    if job.sampler.transform != nothing
      job.sstate.pstate.tensorlogtarget[:, :] = job.sampler.transform(job.sstate.pstate.tensorlogtarget)
    end

    job.sstate.newinvtensor[:, :] = inv(job.sstate.pstate.tensorlogtarget)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

    job.sstate.ratio += (
      0.5*(
        logdet(job.sstate.tune.totaltune.step*job.sstate.oldinvtensor)+dot(
          job.sstate.pstate.value-job.sstate.μ,
          job.pstate.tensorlogtarget*(job.sstate.pstate.value-job.sstate.μ)
        )/job.sstate.tune.totaltune.step
      )
    )

    job.sstate.newfirstterm[:] = job.sstate.newinvtensor*job.sstate.pstate.gradlogtarget

    job.sstate.μ[:] = job.sstate.pstate.value+0.5*job.sstate.tune.totaltune.step*job.sstate.newfirstterm

    job.sstate.ratio -= (
      0.5*(
        logdet(job.sstate.tune.totaltune.step*job.sstate.newinvtensor)+dot(
          job.pstate.value-job.sstate.μ,
          job.sstate.pstate.tensorlogtarget*(job.pstate.value-job.sstate.μ)
        )/_job.sstate.tune.totaltune.step
      )
    )

    if job.sstate.ratio > 0 || (job.sstate.ratio > log(rand()))
      # update
      job.pstate.value = copy(job.sstate.pstate.value)

      job.pstate.gradlogtarget = copy(job.sstate.pstate.gradlogtarget)

      if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
        job.pstate.gradloglikelihood = copy(job.sstate.pstate.gradloglikelihood)
      end

      if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
        job.pstate.gradlogprior = copy(job.sstate.pstate.gradlogprior)
      end

      job.pstate.tensorlogtarget = copy(job.sstate.pstate.tensorlogtarget)

      if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
        job.pstate.tensorloglikelihood = copy(job.sstate.pstate.tensorloglikelihood)
      end

      if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
        job.pstate.tensorlogprior = copy(job.sstate.pstate.tensorlogprior)
      end

      job.sstate.oldinvtensor = copy(job.sstate.newinvtensor)

      job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(job.sstate.newinvtensor)))

      job.sstate.oldfirstterm = copy(job.sstate.newfirstterm)

      job.pstate.logtarget = job.sstate.pstate.logtarget

      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
      end

      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstate.logprior
      end

      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
        end
      end

      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        job.sstate.tune.totaltune.accepted += 1
      end

      if job.tuner.smmalatuner.verbose
        job.sstate.tune.smmalatune.accepted += 1
      end
    else
      # noupdate
      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
        end
      end
    end
  else
    # ambody
    if job.tuner.amtuner.verbose
      job.sstate.tune.amtune.proposed += 1
    end

    covariance!(
      job.sstate.oldinvtensor,
      job.sstate.oldinvtensor,
      job.sstate.count-2,
      job.pstate.value,
      job.sstate.lastmean,
      job.sstate.secondlastmean
    )

    job.sstate.oldinvtensor[:, :] = Hermitian(job.sstate.oldinvtensor)

    set_gmm!(job.sstate, job.sampler, job.pstate)

    job.sstate.pstate.value[:] =  rand(job.sstate.proposal)

    job.parameter.logtarget!(job.sstate.pstate)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

    job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)

    set_gmm!(job.sstate, job.sampler, job.sstate.pstate)

    job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)

    if job.sstate.ratio > 0 || (job.sstate.ratio > log(rand()))
      # update
      job.pstate.value = copy(job.sstate.pstate.value)

      job.pstate.logtarget = job.sstate.pstate.logtarget

      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
      end

      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstate.logprior
      end

      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
        end
      end

      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        job.sstate.tune.totaltune.accepted += 1
      end

      if job.tuner.amtuner.verbose
        job.sstate.tune.amtune.accepted += 1
      end
    else
      # noupdate
      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
        end
      end
    end
  end

  job.sstate.secondlastmean = copy(job.sstate.lastmean)

  recursive_mean!(job.sstate.lastmean, job.sstate.lastmean, job.sstate.count, job.pstate.value)

  job.sstate.pastupdatetensor = job.sstate.presentupdatetensor

  if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune.totaltune)))
  end

  if job.tuner.smmalatuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.smmalafrequency = _job.sstate.tune.smmalatune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.amtuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.amfrequency = _job.sstate.tune.amtune.proposed/_job.sstate.tune.totaltune.proposed)
    )
    push!(burninbody, :(rate!(_job.sstate.tune.amtune)))
  end

  if job.tuner.totaltuner.verbose || job.tuner.smmalatuner.verbose || job.tuner.amtuner.verbose
    fmt_tot_iter = format_iteration(ndigits(job.range.burnin))
    fmt_burnin_iter = format_iteration(ndigits(job.tuner.totaltuner.period))
    fmt_perc = format_percentage()

    push!(burninbody, :(println(
      "Burnin iteration ",
      $(fmt_tot_iter)(_job.sstate.count),
      " out of ",
      _job.range.burnin,
      "..."
    )))

    if job.tuner.totaltuner.verbose
      push!(burninbody, :(println(
        "  Total : ",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.totaltune.rate),
        "%) acceptance rate"
      )))
    end

    if job.tuner.smmalatuner.verbose
      push!(burninbody, :(println(
        "  SMMALA: ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalatune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.smmalatune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.smmalafrequency),
        "%) running frequency"
      )))
    end

    if job.tuner.amtuner.verbose
      push!(burninbody, :(println(
        "  AM    : ",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.accepted),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.amtune.rate),
        "%) acceptance rate, ",
        $(fmt_burnin_iter)(_job.sstate.tune.amtune.proposed),
        "/",
        $(fmt_burnin_iter)(_job.sstate.tune.totaltune.proposed),
        " (",
        $(fmt_perc)(100*_job.sstate.tune.amfrequency),
        "%) running frequency"
      )))
    end
  end

  if isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune.totaltune, _job.tuner.totaltuner)))
    push!(burninbody, :(_job.sstate.sqrttunestep = sqrt(_job.sstate.tune.totaltune.step)))
  end

  if job.tuner.smmalatuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.amtuner.verbose
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.amtune)))
  end

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    push!(burninbody, :(_job.sstate.tune.totaltune.totproposed += _job.sstate.tune.totaltune.proposed))

    push!(burninbody, :(_job.sstate.tune.totaltune.proposed = 0))

    if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
      push!(burninbody, :(_job.sstate.tune.totaltune.accepted = 0))
      push!(burninbody, :(_job.sstate.tune.totaltune.rate = NaN))
    end

    push!(
      body,
      Expr(
        :if,
        :(
          _job.sstate.tune.totaltune.totproposed <= _job.range.burnin &&
          mod(_job.sstate.tune.totaltune.proposed, _job.tuner.totaltuner.period) == 0
        ),
        Expr(:block, burninbody...)
      )
    )
  end

  if !job.plain
    push!(body, :(produce()))
  end
end
