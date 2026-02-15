
begin
	xy_vpol_p1 = [(x, y) for x in -1.5:1.5 for y in [11]]
	xy_vpol_p2 = [(x, y) for x in -2.5:2.5 for y in [10]]
	xy_vpol_p3 = [(x, y) for x in -2.5:2.5 for y in [9]]
	xy_vpol_p4 = [(x, y) for x in -3.5:3.5 for y in [8]]

	xy_vpol_p5 = [(x, y) for x in -7.5:-4.5 for y in [7]]
	xy_vpol_p6 = [(x, y) for x in -2.5:2.5 for y in [7]]
	xy_vpol_p7 = [(x, y) for x in 4.5:7.5 for y in [7]]

	xy_vpol_p8 = [(x, y) for x in -8.5:8.5 for y in [6]]

	xy_vpol_p9 = [(x, y) for x in -8.5:-3.5 for y in [5]]
	xy_vpol_p10 = [(x, y) for x in -1.5:1.5 for y in [5]]
	xy_vpol_p11 = [(x, y) for x in 3.5:8.5 for y in [5]]

	xy_vpol_p12 = [(x, y) for x in -9.5:-2.5 for y in [4]]
	xy_vpol_p13 = [(x, y) for x in 2.5:9.5 for y in [4]]

	xy_vpol_p14 = [(x, y) for x in -8.5:-3.5 for y in [3]]
	xy_vpol_p15 = [(x, y) for x in -1.5:1.5 for y in [3]]
	xy_vpol_p16 = [(x, y) for x in 3.5:8.5 for y in [3]]

	xy_vpol_p17 = [(x, y) for x in -8.5:8.5 for y in [2]]

	xy_vpol_p18 = [(x, y) for x in -7.5:-4.5 for y in [1]]
	xy_vpol_p19 = [(x, y) for x in -2.5:2.5 for y in [1]]
	xy_vpol_p20 = [(x, y) for x in 4.5:7.5 for y in [1]]

	xy_vpol_21 = [(x, y) for x in -3.5:3.5 for y in [0]]
end

xy_vpol_p_all = vcat([getfield(@__MODULE__, Symbol("xy_vpol_p$i")) for i in 1:20]...)
xy_vpol_n_all = [(s[1], -s[2]) for s in xy_vpol_p_all]
xy_vpol_all = vcat(xy_vpol_p_all, xy_vpol_n_all, xy_vpol_21)


