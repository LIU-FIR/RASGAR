
xy_hpol_p_all = Vector{Tuple{Float64, Float64}}[]
begin
	push!(xy_hpol_p_all, [(x, y) for x in -1:1 for y in [11.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [10.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [9.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -3:3 for y in [8.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -7:-5 for y in [7.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -3:3 for y in [7.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 5:7 for y in [7.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -8:-4 for y in [6.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [6.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 4:8 for y in [6.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -8:-4 for y in [5.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [5.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 4:8 for y in [5.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -9:-3 for y in [4.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -1:1 for y in [4.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 3:9 for y in [4.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -9:-3 for y in [3.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -1:1 for y in [3.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 3:9 for y in [3.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -8:-4 for y in [2.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [2.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 4:8 for y in [2.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -8:-4 for y in [1.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -2:2 for y in [1.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 4:8 for y in [1.5]])

	push!(xy_hpol_p_all, [(x, y) for x in -7:-5 for y in [0.5]])
	push!(xy_hpol_p_all, [(x, y) for x in -3:3 for y in [0.5]])
	push!(xy_hpol_p_all, [(x, y) for x in 5:7 for y in [0.5]])
end

xy_hpol_p_all = vcat(xy_hpol_p_all...)

xy_hpol_n_all = [(s[1], -s[2]) for s in xy_hpol_p_all]
xy_hpol_all = vcat(xy_hpol_p_all, xy_hpol_n_all)
xy_hpol_all = [(Float64(v[1]), Float64(v[2])) for v in xy_hpol_all]
xy_hpol_all

