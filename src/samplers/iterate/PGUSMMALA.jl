function codegen(::Type{Val{:iterate}}, ::Type{PGUSMMALA}, job::BasicMCJob)
  result::Expr
  update = []
  noupdate = []
  burninbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in PGUSMMALA code generation")
  end

  if job.tuner.verbose
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  push!(body, :(
    _job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.oldfirstterm)
  )

  push!(
    body,
    :(
      if _job.sstate.updatetensor
        _job.sstate.cholinvtensor = chol(_job.sstate.oldinvtensor, Val{:L})
      end
    )
  )

  push!(
    body,
    :(_job.sstate.pstate.value = _job.sstate.μ+sqrt(_job.sstate.tune.step)*_job.sstate.cholinvtensor*randn(_job.pstate.size))
  )

  push!(body, :(_job.sampler.update!(_job.sstate, _job.pstate)))

  smmalabody = [:(_job.sstate.updatetensorcount += 1)]

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.pstate)))

  push!(smmalabody, :(_job.sstate.oldinvtensor = inv(_job.pstate.tensorlogtarget)))

  push!(smmalabody, :(_job.sstate.cholinvtensor = chol(_job.sstate.oldinvtensor, Val{:L})))

  push!(smmalabody, :(_job.sstate.oldfirstterm = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget))

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.sstate.pstate)))

  if job.sampler.transform != nothing
    push!(smmalabody, :(_job.sstate.pstate.tensorlogtarget = _job.sampler.transform(_job.sstate.pstate.tensorlogtarget)))
  end

  push!(smmalabody, :(_job.sstate.newinvtensor = inv(_job.sstate.pstate.tensorlogtarget)))

  malabody = [:(_job.parameter.uptogradlogtarget!(_job.sstate.pstate))]

  push!(body, Expr(:if, :(_job.sstate.updatetensor), Expr(:block, smmalabody...), malabody...))

  push!(body, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    body,
    :(
      _job.sstate.ratio += (
        sum(log(diag(sqrt(_job.sstate.tune.step)*_job.sstate.cholinvtensor)))
        +0.5*dot(
          _job.sstate.pstate.value-_job.sstate.μ,
          _job.pstate.tensorlogtarget*(_job.sstate.pstate.value-_job.sstate.μ)
        )/_job.sstate.tune.step
      )
    )
  )

  push!(body, :(_job.sstate.newfirstterm = _job.sstate.newinvtensor*_job.sstate.pstate.gradlogtarget))

  push!(body, :(_job.sstate.μ = _job.sstate.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.newfirstterm))

  smmalabody = [
    :(_job.sstate.cholinvtensor = chol(_job.sstate.newinvtensor, Val{:L})),
    :(
      _job.sstate.ratio -= (
        sum(log(diag(sqrt(_job.sstate.tune.step)*_job.sstate.cholinvtensor)))
        +0.5*dot(
          _job.pstate.value-_job.sstate.μ,
          _job.sstate.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
        )/_job.sstate.tune.step
      )
    )
  ]

  malabody = [
    :(
      _job.sstate.ratio -= (
        sum(log(diag(sqrt(_job.sstate.tune.step)*_job.sstate.cholinvtensor)))
        +0.5*dot(
          _job.pstate.value-_job.sstate.μ,
          _job.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
        )/_job.sstate.tune.step
      )
    )
  ]

  push!(body, Expr(:if, :(_job.sstate.updatetensor), Expr(:block, smmalabody...), malabody...))

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))

  if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
  end

  if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
  end

  smmalabody = [:(_job.pstate.tensorlogtarget = copy(_job.sstate.pstate.tensorlogtarget))]

  if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
    push!(smmalabody, :(_job.pstate.tensorloglikelihood = copy(_job.sstate.pstate.tensorloglikelihood)))
  end

  if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
    push!(smmalabody, :(_job.pstate.tensorlogprior = copy(_job.sstate.pstate.tensorlogprior)))
  end

  push!(update, Expr(:if, :(_job.sstate.updatetensor), Expr(:block, smmalabody...)))

  push!(update, :(_job.sstate.oldinvtensor = copy(_job.sstate.newinvtensor)))

  push!(update, :(_job.sstate.oldfirstterm = copy(_job.sstate.newfirstterm)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end

  if job.tuner.verbose
    push!(update, :(_job.sstate.tune.accepted += 1))
  end

  push!(body, Expr(:if, :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))), Expr(:block, update...), noupdate...))

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune)))

    if isa(job.tuner, AcceptanceRateMCTuner)
      push!(burninbody, :(tune!(_job.sstate.tune, _job.tuner)))
    end

    if job.tuner.verbose
      fmt_iter = format_iteration(ndigits(job.range.burnin))
      fmt_perc = format_percentage()

      push!(burninbody, :(println(
        "Burnin iteration ",
        $(fmt_iter)(_job.sstate.tune.totproposed),
        " of ",
        _job.range.burnin,
        ": ",
        $(fmt_perc)(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )))
    end

    push!(burninbody, :(reset_burnin!(_job.sstate.tune)))

    push!(body, Expr(
      :if,
      :(_job.sstate.tune.totproposed <= _job.range.burnin && mod(_job.sstate.tune.proposed, _job.tuner.period) == 0),
      Expr(:block, burninbody...)
    ))
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
