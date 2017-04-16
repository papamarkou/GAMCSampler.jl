exp_decay(i::Integer, tot::Integer, a::Real=10., b::Real=0.) = (1-b)*exp(-a*i/tot)+b

lin_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*i))+b

pow_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(a^(i/tot))+b

quad_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*abs2(i)))+b
