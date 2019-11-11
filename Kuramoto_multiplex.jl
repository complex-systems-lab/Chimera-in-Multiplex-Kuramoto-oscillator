using DifferentialEquations
using Random
using PyPlot
using Printf
using Statistics
using DataFrames
using IterableTables
using LinearAlgebra
using DelimitedFiles
using CSV
using LinearAlgebra

################Regular Network G generating function###############################
function regular(n::Integer,k::Integer)
    !iseven(n*k) && throw(ArgumentError("n*k must be even!"))
    !(0<=k<n) &&  throw(ArgumentError("The 0<= k<n must be satisfied!!"))

    G=zeros((n,n))
    for i=1:n-1
        for j=i+1:n
            if(abs(i-j)<=k/2)
                G[i,j]=G[j,i]=1
            else
                if((n-abs(i-j))<=abs(k/2))
                    G[i,j]=G[j,i]=1
                end
            end
        end
    end

    return G

end

########################################################################
#Input Parameters
global Nl=250                                    #Network Size
global r=0.32                                    #Coupling Radius
global lambda=0.1                                  #Overall Coupling Strength
global omega=0.01*ones(2*Nl)                       #Constant natural frequency for all oscilators
global alpha=1.43                                # Phase lag parameter for chimera state
#######################################################################

######Initial Phases#####################################
#chosen such that they are randomly distributed with an gausian envelop
rng = MersenneTwister(58959);
theta0=[6*(rand(rng)-0.5)*exp(-30*(x/float(Nl)-0.5)*(x/float(Nl)-0.5))  for x in range(1,stop=Nl)]
theta0=vcat(theta0,theta0)
writedlm("generated_initial.txt", theta0)
plot(theta0,"go",markersize=2);
savefig("Initial_condition.png")
##################################################################
kk = floor(Int64,r*Nl*2)
A=regular(Nl,kk);
Id=Matrix(1.0I, Nl, Nl);
MA=vcat(hcat(A,Id),hcat(Id,A));
clf()
imshow(MA,cmap="hot")
savefig("Generated_Multiplex.png")
#######################################################################
niter=50000                 # Total Iteration
novp=2000                   # Ovserved Time period after transient
dt = 0.01                   # initial time step size
ti = 0.0; tf = niter*dt
tr = (niter-novp)*dt
tspan = (ti, tf)             # Time interval
#######################################################################
function kura2(du,u,p,t)
    global omega, A, m, lambda , Nl
    n = floor(Int64, length(u))
    for i = 1:n
        du[i] = omega[i] + (lambda/float((r*Nl*2)+1) * (sum(j->(MA[i,j]* sin(u[j]-u[i]+alpha)),1:2*Nl))   ) 
    end
end
##############################################################################
function order_para(y)
    n = floor(Int64,length(y))
    real_sum = 0.0; imag_sum = 0.0
    for l = 1:n
        real_sum += cos(y[l])
        imag_sum += sin(y[l])
    end
    real_sum = real_sum/n
    imag_sum = imag_sum/n
    r = sqrt((real_sum)^2 + (imag_sum)^2)
    return r
end
############################################################################
#system integration
prob = ODEProblem(kura2,theta0,tspan)
sol = solve(prob,Tsit5(), reltol=1e-6, saveat=dt) ;
###################################################################
clf()
x= [mod2pi(i) for i in sol[:,niter]];
ylim([0,2*pi])
writedlm("Final_state.txt",x)
title("Phase Plot")
plot(x,"mo",markersize=2)
savefig("Final_state.png")
########################Frequency Calculation#############################################
T=size(sol)[2]
n=size(sol)[1]
x = zeros(0)

for y=1:n
    sigp=[mod2pi(sol[Int(y),t])-pi for t in range(niter-10000,stop=niter)]
    upcross=findall((sigp[1:end-1] .<=0) .& (sigp[2:end] .>0))
    ff=2*pi*size(upcross)[1]/((upcross[end] - upcross[1])*0.01)
    append!( x, ff ) 
end
clf()
title("Frequency Plot")
plot(x,"ro",markersize=2)
savefig("Final_Frequency.png")

