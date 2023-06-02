using LinearAlgebra
using LaTeXStrings
using PGFPlotsX
using JLD
using HDF5



figures_phase = Array{PGFPlotsX.Axis}(undef, 4)
snrs = [10 20 30 40]
for i = 1:4
    snr = snrs[i]
    ts = load("./noisy/inputdata/$(snr)_t_series.jld","t_series")
    figures_phase[i] = @pgf Axis(
        {
            xlabel = L"$y_{w1}$(mm)",
            ylabel = L"$\varphi_{w1}$",
            legend_pos = "north west",
            height = "8cm",
            width = "8cm",
            xmin = -20,
            xmax = 20,
            ymin = -20,
            ymax = 20,
            ylabel_shift="-10pt",
            xlabel_shift="-3pt",
            title="SNR=$(snr)db"
        },
        Plot({color = "blue", no_marks}, Coordinates(ts[1][1, :], ts[1][2, :])),
        LegendEntry("Phase figure"),
    )

end

g = @pgf GroupPlot(
{ group_style = { group_size="4 by 1" },
  no_markers
},
figures_phase[1],figures_phase[2],figures_phase[3],figures_phase[4])
pgfsave("./noisy/inputdata/noise_phase.pdf",g)



figures_ts = Array{PGFPlotsX.Axis}(undef, 4)
snrs = [10 20 30 40]
for i = 1:4
    snr = snrs[i]
    ts = load("./noisy/inputdata/$(snr)_t_series.jld","t_series")
    point_num = 200
    len = length(ts[1][1,:])
    begin_index = len - point_num-1
    figures_ts[i] = @pgf Axis(
        {
            xlabel = "time(s)",
            ylabel = L"$y_{w1}(mm)$",
            legend_pos = "north west",
            height = "8cm",
            width = "8cm",
            xmin = begin_index,
            ymin = -15,
            ymax = 15,
            ylabel_shift="-10pt",
            xlabel_shift="-3pt",
            title="SNR=$(snr)db"
        },
        Plot({color = "blue", no_marks}, Coordinates(begin_index+1:begin_index+point_num, ts[1][1, end-point_num+1:end])),
        LegendEntry("Time series figure"),
    )
end

t = @pgf GroupPlot(
{ group_style = { group_size="4 by 1" },
  no_markers
},
figures_ts[1],figures_ts[2],figures_ts[3],figures_ts[4])
pgfsave("./noisy/inputdata/noise_time.pdf",t)
