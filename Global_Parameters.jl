module Global_Parameters

export nh,V₀,vel_l,Vel,Vel2,SNRs

##basic parameters
#Number of Fourier expansion terms
nh = 20

#Predicted bifurcation point;Precise value:97.67
V₀ = 98

#Number and range of samples
vel_l = 8
Vel = range(98; stop=110, length=vel_l)

Vel2 = 97:0.1:112

#signal-to-noise ratio
SNRs = [10 15 20 30 40]

vdp_Vel = range(0.01; stop=0.02, length=vel_l)
vdp_Vel2 = 0.01:0.0005:0.02

hunting_Vel = range(0.001; stop=0.02, length=vel_l)
hunting_Vel2 = range(0.001; stop=0.02, length=30)

normal_digits = 6
end
