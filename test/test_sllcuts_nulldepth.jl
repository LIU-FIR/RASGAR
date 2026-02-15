# ========================================================
# /test/test_sllcuts_nulldepth.jl
# Unit tests for sll_1dcut / sll_uv_cuts / null_depth_db
# ========================================================

using Test
using LinearAlgebra
using Random

# Load dev module (keeps tests runnable without package install)
include(joinpath(@__DIR__, "..", "src", "RASGAR.jl"))
using .RASGAR

# test passed
@testset "PattnEval.sll_1dcut" begin
	# axis in uv domain
	axis = collect(range(-0.6, 0.6; length = 241))

	# Construct a synthetic normalized power cut:
	# - mainlobe: narrow Gaussian centered at 0, peak 1
	# - sidelobe: bump at u=0.30, peak 0.10
	# - secondary sidelobe: bump at u=-0.42, peak 0.07
	σ     = 0.04
	g_main = @. exp(-(axis/σ)^2)
	g_sl1  = @. 0.10 * exp(-((axis-0.30)/0.02)^2)
	g_sl2  = @. 0.07 * exp(-((axis+0.42)/0.02)^2)
	g      = g_main .+ g_sl1 .+ g_sl2
	g      ./= maximum(g)  # normalize to peak=1

	# Use a reasonable HPBW in axis units (uv). Any positive value works as long
	# as mainlobe exclusion window doesn't cover the sidelobe bumps.

	hpbw = 2σ*sqrt(log(2))
	out = RASGAR.PattnEval.sll_1dcut(axis, g; hpbw = hpbw, k = 2.5, center = 0.0)

	@test isfinite(out.sll_db)
	@test out.idx != 0

	println("out.loc: ", out.loc)
	println("out.sll_db: ", out.sll_db)
	@test isapprox(out.loc, 0.30; atol = 0.01) # test passed
	@test isapprox(out.sll_db, 10*log10(0.10); atol = 0.6) # test passed
end

# test passed
@testset "PattnEval.sll_uv_cuts (PattNT U/V vectors)" begin
	# U/V vectors and a 2D power map with controlled sidelobes.
	U = collect(range(-0.6, 0.6; length = 241))
	V = collect(range(-0.5, 0.5; length = 201))
	Nu, Nv = length(U), length(V)

	σu, σv = 0.04, 0.05
	G = Array{Float64}(undef, Nu, Nv)

	# mainlobe (power) centered at (0,0), peak ~1
	@inbounds for j in 1:Nv
		v = V[j]
		for i in 1:Nu
			u = U[i]
			G[i, j] = exp(-((u/σu)^2 + (v/σv)^2))
		end
	end

	# Add sidelobe bumps:
	# - along u-cut at v≈0: bump at u=0.35 with power 0.10
	# - along v-cut at u≈0: bump at v=-0.25 with power 0.08
	for j in 1:Nv
		v = V[j]
		for i in 1:Nu
			u = U[i]
			G[i, j] += 0.10 * exp(-((u-0.35)/0.02)^2 - (v/0.03)^2)
			G[i, j] += 0.08 * exp(-(u/0.03)^2 - ((v+0.25)/0.02)^2)
		end
	end

	patt = (U = U, V = V, G = G)
	pk = RASGAR.PattnEval.peak_uv(patt)
	@test isapprox(pk.u, 0.0; atol = 1e-3)
	@test isapprox(pk.v, 0.0; atol = 1e-3)

	# sll_uv_cuts yields hpbw_u/hpbw_v slightly smaller than 2σu/2σv
	# To make win_half=k*hpbw/2 approximate [-3σ, 3σ], increase k to expand main lobe protection zone
	# k_tst = 2.5
	# Use sll_uv_cuts, which will compute HPBW from the u/v cuts.
	res = RASGAR.PattnEval.sll_uv_cuts(patt; k = 2.5)

	@test isfinite(res.u.sll_db)
	@test isfinite(res.v.sll_db)

	# sidelobe locations near the injected bumps
	@test isapprox(res.u.loc, 0.35; atol = 0.02) # test failed
	@test isapprox(res.v.loc, -0.25; atol = 0.02) # test failed

	# sidelobe levels (relative to peak power)
	@test isapprox(res.u.sll_db, 10*log10(0.10); atol = 1.0) # test failed
	@test isapprox(res.v.sll_db, 10*log10(0.08); atol = 1.0) # test failed
end

# test passed
@testset "PattnEval.null_depth_db" begin
	# Build a tiny array model; we only need steer_vec to work.
	xy = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
	M = length(xy)
	geom = RASGAR.RasSpecs.CustomXYGeomSpec(
		xy = xy,
		unit = :grid,
		d = 0.5,
		z0 = 0.0,
		center = :none,
	)
	calib = RASGAR.RasSpecs.ElemGainCalibSpec(g = ones(ComplexF64, M))
	rarray_spec = RASGAR.RasSpecs.RArraySpec(
		geom            = geom,
		calib_reference = calib,
		calib_assumed   = calib,
		mask            = nothing,
	)
	rarray = RASGAR.RasModels.build_rarray(rarray_spec)

	λ = 0.21428571428571427  # ~ c / 1.4GHz (exact value not critical for test)
	mode = :assumed

	# Choose a look direction and an interference (null) direction in uv.
	us, vs = 0.0, 0.0
	ui, vi = 0.30, -0.10

	aS = RASGAR.RArrCores.steer_vec(us, vs; rarray = rarray, λ = λ, mode = mode)
	aI = RASGAR.RArrCores.steer_vec(ui, vi; rarray = rarray, λ = λ, mode = mode)

	# Construct w that has an exact (numerical) null at (ui,vi):
	# w = aS - proj_{aI}(aS)
	α = dot(aI, aS) / (dot(aI, aI) + eps(Float64))
	w = aS .- α .* aI

	# Mainlobe peak power at steering direction
	gmax = abs2(dot(w, aS))
	@test isfinite(gmax) && gmax > 0

	nd = RASGAR.PattnEval.null_depth_db(w; rarray = rarray, λ = λ, ui = ui, vi = vi, gmax = gmax, mode = mode)
	# nd might underflow to -Inf if the null is numerically exact.
	@test (isfinite(nd) && nd < -80.0) || (nd == -Inf)
end
