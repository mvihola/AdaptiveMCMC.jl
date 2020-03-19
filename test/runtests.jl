using Test
using Random
using Statistics
using AdaptiveMCMC

Random.seed!(1234)

# Standard normal test target in 2d
log_p(x) = -(x[1]^2+x[2]^2)/2
x0 = zeros(2)
n = 100_000
test_means(X, atol=0.05) = all(abs.(mapslices(mean, X, dims=2)) .<= atol)
test_vars(X, atol=0.05) = all(abs.(mapslices(var, X, dims=2) .- 1.0) .<= atol)
test_stats(X, atol=0.05) = test_vars(X, atol) && test_means(X, atol)
test_acc(acc, acc_opt=0.234, atol=0.05) = all(abs.(acc .- acc_opt) .<= atol)

o_ram = adaptive_rwm(x0, log_p, n; algorithm=:ram)
@test test_acc(o_ram.accRWM)
@test test_stats(o_ram.X)
o_ram_32 = adaptive_rwm(Float32.(x0), log_p, n; algorithm=:ram)
@test test_acc(o_ram_32.accRWM)
@test test_stats(o_ram_32.X)

o_am = adaptive_rwm(x0, log_p, n; algorithm=:am)
@test o_am.S[1].L.L[1] ≈ 1.0 atol=0.05
@test test_stats(o_am.X)
o_am_32 = adaptive_rwm(x0, log_p, n; algorithm=:am)
@test o_am_32.S[1].L.L[1] ≈ 1.0 atol=0.05
@test test_stats(o_am_32.X)


o_asm = adaptive_rwm(x0, log_p, n; algorithm=:asm)
@test test_acc(o_asm.accRWM)
@test test_stats(o_asm.X)
o_asm_32 = adaptive_rwm(Float32.(x0), log_p, n; algorithm=:asm)
@test test_acc(o_asm_32.accRWM)
@test test_stats(o_asm_32.X)

o_aswam = adaptive_rwm(x0, log_p, n; algorithm=:aswam)
@test o_aswam.S[1].AM.L.L[1] ≈ 1.0 atol=0.05
@test test_acc(o_aswam.accRWM)
@test test_stats(o_aswam.X)
o_aswam_32 = adaptive_rwm(Float32.(x0), log_p, n; algorithm=:aswam)
@test o_aswam_32.S[1].AM.L.L[1] ≈ 1.0 atol=0.10
@test test_acc(o_aswam_32.accRWM)
@test test_stats(o_aswam_32.X)

o_apt = adaptive_rwm(x0, log_p, n; L=3, all_levels=true)
@test test_acc(o_apt.accRWM)
@test test_acc(o_apt.accSW)
@test test_stats(o_apt.X)
@test test_means(o_apt.allX[2], 0.10)
@test test_means(o_apt.allX[3], 0.15)
o_apt_32 = adaptive_rwm(Float32.(x0), log_p, n; L=3, all_levels=true)
@test test_acc(o_apt_32.accRWM)
@test test_acc(o_apt_32.accSW)
@test test_stats(o_apt_32.X)
@test test_means(o_apt_32.allX[2], 0.10)
@test test_means(o_apt_32.allX[3], 0.15)
