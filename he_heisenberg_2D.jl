PKG_METEOR_PATH = "/home/ubuntu/quantum_adaboost/Meteor/src"
push!(LOAD_PATH, PKG_METEOR_PATH)
using Meteor.QuantumCircuit
using Meteor.Diff
using Flux
using LinearAlgebra
using Random
using Zygote: @adjoint
using SparseArrays


function creat_graph()
    #create graph of the Heisenberg model
    A=zeros(Int,8,8)
    global Edge=[]
    push!(Edge,[1,2])
    push!(Edge,[1,3])
    push!(Edge,[1,4])
    push!(Edge,[1,5])
    push!(Edge,[5,6])
    push!(Edge,[6,7])
    push!(Edge,[2,3])
    push!(Edge,[2,8])

    A[1,2]=1
    A[1,3]=1
    A[1,4]=1
    A[1,5]=1
    A[5,6]=1
    A[6,7]=1
    A[2,3]=1
    A[2,8]=1
    # A[4,6]=1
    # A[1,3]=1
    A0=sparse(A)
    return A0
end

function afHeisenberg(A0)
    #Build a Heisenberg model through the graph
    J=1
    Jzz=0.5
    II, JJ, V=findnz(A0)
    ham = QubitsOperator()
    for c =1:length(II)
        ham += QubitsTerm(II[c]=>"X",JJ[c]=>"X", coeff=J*V[c])
        ham += QubitsTerm(II[c]=>"Y",JJ[c]=>"Y", coeff=J*V[c])
        ham += QubitsTerm(II[c]=>"Z",JJ[c]=>"Z", coeff=J*Jzz*V[c])
    end
    htot=3*sum(V)
    Ham=matrix(ham)
    Ham0=Array(Ham)
    r=abs.(eigvals(Ham0))
    r0=findmax(r)[1]
    J=J
    htot=htot/r0
    ham0 = QubitsOperator()
    for c =1:length(II)
        ham0 += QubitsTerm(II[c]=>"X",JJ[c]=>"X", coeff=J*V[c])
        ham0 += QubitsTerm(II[c]=>"Y",JJ[c]=>"Y", coeff=J*V[c])
        ham0 += QubitsTerm(II[c]=>"Z",JJ[c]=>"Z", coeff=J*Jzz*V[c])
    end
    return ham0,J,htot
end


function Heisenberg_2D(nrow::Int,ncol::Int, J::Real, Jzz::Real)
    n = nrow * ncol
    idx = [1:n...]
    sites = reshape(idx, (nrow, ncol))

    H = QubitsOperator()

    for i = 1:nrow-1
        for j = 1:ncol
            idx1 = sites[i,j]
            idx2 = sites[i+1,j]
            H += QubitsTerm(idx1=>"X", idx2=>"X", coeff=J)
            H += QubitsTerm(idx1=>"Y", idx2=>"Y", coeff=J)
            H += QubitsTerm(idx1=>"Z", idx2=>"Z", coeff=J*Jzz)
        end
    end

    for j = 1:nrow
        for i = 1:ncol-1
            idx1 = sites[j,i]
            idx2 = sites[j,i+1]
            H += QubitsTerm(idx1=>"X", idx2=>"X", coeff=J)
            H += QubitsTerm(idx1=>"Y", idx2=>"Y", coeff=J)
            H += QubitsTerm(idx1=>"Z", idx2=>"Z", coeff=J*Jzz)
        end
    end

    return  H
end

function generatepsi(num)
    #generate initial state
   ϕ=(statevector([1,0])-statevector([0,1]))./sqrt(2)
   psi=ϕ
   if num>2
       for i=2:num/2
           psi=QuantumCircuit.qcat(psi, ϕ)
       end
   end
   return psi
end

function HEA(params,L::Int,nlayers::Int)
    circuit = QCircuit()
    counts=0

    for j=1:nlayers
        for i = 1:L
            counts += 1
            push!(circuit, RzGate(i, Variable(params[counts])))
            counts += 1
            push!(circuit, RyGate(i, Variable(params[counts])))
        end

        for i=1:2:L-1
            push!(circuit, CNOTGate(i,i+1))
        end

        for i=2:2:L-1
            push!(circuit, CNOTGate(i,i+1))
        end
    end

    for i = 1:L
        counts += 1
        push!(circuit, RzGate(i, Variable(params[counts])))
        counts += 1
        push!(circuit, RyGate(i, Variable(params[counts])))
    end
   
    return circuit
end

function build_toy_Ansatz(params)#,L::Int,nlayers::Int
    
    circuit = QCircuit()
    circuit = HEA(params,L::Int,nlayers::Int)

    return circuit
end

build_Ansatz(params)=build_toy_Ansatz(params)
@adjoint build_toy_Ansatz(params) =build_toy_Ansatz(params), z -> (z,)
loss(params) = real(expectation(ham, build_Ansatz(params) * ψ ))

function build_model(lays1,lays2,layer3)
    #Neural network with hidden layer 1
    return Chain( Dense(lays1=>lays2, relu),
    Dense(lays2=>layer3, relu)
                    )
end
function build_model2(lays1,lays2,layer3,layer4)
    #Neural network with hidden layer 2
    return Chain( Dense(lays1=>lays2),
    Dense(lays2=>layer3, relu),
    Dense(layer3=>layer4, relu)
                    )
end
function build_model3(lays1,lays2,layer3,layer4,layer5)
    #Neural network with hidden layer 2
    return Chain( Dense(lays1=>lays2,sigmoid),
    Dense(lays2=>layer3, sigmoid),
    Dense(layer3=>layer4, sigmoid),
    Dense(layer4=>layer5,sigmoid)
                    )
end
function build_model4(lays1,lays2,layer3,layer4,layer5,layer6)
    #Neural network with hidden layer 2
    return Chain( Dense(lays1=>lays2),
    Dense(lays2=>layer3, relu),
    Dense(layer3=>layer4, relu),
    Dense(layer4=>layer5, sigmoid),
    Dense(layer5=>layer6)
                    )
end
function generPsi_x(THETA)
    #100 quantum states of phi are generated from 100 theta parameters
    a=length(THETA[1,:])
    Psi=[]
    circuit=build_Ansatz(THETA[:,1])
    for i in 1:a
        set_parameters!(THETA[:,i],circuit)
        x=circuit *ψ
        push!(Psi,x)
    end
    return Psi
end

function gener_theta()

    result = zeros(Float64, Npara, num_samples)

    for i in 1:num_samples
        for j in 1:Npara
            result[j, i] = rand() * 2π
        end
    end

    return result
end

function loss0(falpha0)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    # alpha=falpha0[1,:].*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    cc=sum(abs.(alpha))
    # deno=real(alpha_Psi'*alpha_Psi)/(cc^2)
    avg_e=real((alpha_Psi'*Ham*alpha_Psi)/(alpha_Psi'*alpha_Psi))                                                                                                                                                  
    return avg_e[1]
end
function loss2(falpha0,De)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    # alpha=falpha0[1,:].*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    avg_e1=real((alpha_Psi'*Ham*alpha_Psi)/(alpha_Psi'*alpha_Psi))                                                                                                                                      
    println(real(avg_e1))              
    return real(avg_e1)[1]
end


function train_loss_de(t_max,lr_max,lr_min,De)
    i=1
    
    print(i," ")
    # loss3(model(THETA),De)
    println("##############")
    # println(f,"##############")
    i=1
    while i<=t_max||(loss0(model(THETA))>=-0.99)
        print(i," ")
        if i<=t_max
            lr=lr_min+0.5*(lr_max-lr_min)*(1+cos(pi*i/t_max))
        else
            lr=lr_min
        end
        opt =ADAM(lr)
        gs = gradient(() -> loss2(model(THETA),De), ps)
        Flux.Optimise.update!(opt, ps, gs)
        i=i+1
        if i>100000
            break
        end
    end
    flush(stdout)

end

nrow=2
ncol=3
global L=ncol*nrow
J=1
Jzz=0.5
ham = Heisenberg_2D(nrow, ncol, J, Jzz)
# A0=creat_graph()
# II, JJ, V=findnz(A0)
# ham,J,htot=afHeisenberg(A0)

E0,phi = ground_state(ham)
global num_samples = 2000
global Ham=matrix(ham)
global ψ = statevector(ComplexF64, L)
global nlayers=1
global Npara=L*2*(nlayers+1)#The number of variable parameters of a quantum circuit
global model = build_model2( Npara,200,100,2)#neural network model
global THETA=gener_theta()#Generate 100 theta parameter vectors
global Psi_x=generPsi_x(THETA)#100 quantum states of phi are generated from 100 theta parameters
global ps = Flux.params(model)#vectorize the parameters of neural network
#Define the cosin learning rate
global  lr_min=0.00001
global  lr_max=0.0001
global  lr=0.001
global t_max=50000#Defines the maximum number of steps
global De=0.1#Defining the barrier of <1>
println("change loss num_samples = ",num_samples," |0> rand gener, nlayers=",nlayers," L=",L," Npara=",Npara," build_model2( Npara,100,100,2) lrmin= ",lr_min," lr=",lr," lrmax=",lr_max," tmax=",t_max," de>",De)
flush(stdout)
train_loss_de(t_max,lr_max,lr_min,De)#Training parameter