#VARIATIONAL_QUANTUM_CIRCUIT_MONTE_CARLO ALGORITHM
#meteor is a closed source quantum computing package written in the julia programming language for quantum circuit building and gradient differential computing. Our algorithm can be implemented in any quantum computing package.
using Meteor.QuantumCircuit
using Meteor.Diff
using Flux
using LinearAlgebra
using Random
using Zygote: @adjoint
using SparseArrays

function creat_graph()
    #create graph of the Heisenberg model
    A=zeros(Int,6,6)
    global Edge=[]
    push!(Edge,[1,2])
    push!(Edge,[3,4])
    push!(Edge,[3,4])
    push!(Edge,[5,6])
    push!(Edge,[2,1])
    push!(Edge,[3,2])
    push!(Edge,[3,2])
    push!(Edge,[2,6])
    push!(Edge,[4,2])
    push!(Edge,[3,5])
    push!(Edge,[4,6])
    push!(Edge,[1,3])
    A[1,2]=1
    A[3,4]=2
    A[5,6]=1
    A[2,1]=1
    A[3,2]=2
    A[2,6]=1
    A[4,2]=1
    A[3,5]=1
    A[4,6]=1
    A[1,3]=1
    A0=sparse(A)
    return A0
end
function afHeisenberg(A0)
    #Build a Heisenberg model through the graph
    J=1
    II, JJ, V=findnz(A0)
    ham = QubitsOperator()
    for c =1:length(II)
        ham += QubitsTerm(II[c]=>"X",JJ[c]=>"X", coeff=J*V[c])
        ham += QubitsTerm(II[c]=>"Y",JJ[c]=>"Y", coeff=J*V[c])
        ham += QubitsTerm(II[c]=>"Z",JJ[c]=>"Z", coeff=J*V[c])
    end
    htot=3*sum(V)
    Ham=matrix(ham)
    Ham0=Array(Ham)
    r=abs.(eigvals(Ham0))
    r0=findmax(r)[1]
    J=J/r0
    htot=htot/r0
    ham0 = QubitsOperator()
    for c =1:length(II)
        ham0 += QubitsTerm(II[c]=>"X",JJ[c]=>"X", coeff=J*V[c])
        ham0 += QubitsTerm(II[c]=>"Y",JJ[c]=>"Y", coeff=J*V[c])
        ham0 += QubitsTerm(II[c]=>"Z",JJ[c]=>"Z", coeff=J*V[c])
    end
    return ham0,J,htot
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
function build_toy_Ansatz(params)
    #Generated quantum circuit
    circuit = QCircuit()
    counts = 0
    for i=1:Nlayer
        for item in Edge
            #######expzz#####
            counts += 1
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            push!(circuit, RzGate(item[2], Variable(params[counts])))
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            #######expyy#####
            counts += 1
            push!(circuit, RxGate(item[1], -pi/2))
            push!(circuit, RxGate(item[2], -pi/2))
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            push!(circuit, RzGate(item[2], Variable(params[counts])))
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            push!(circuit, RxGate(item[1], pi/2))
            push!(circuit, RxGate(item[2], pi/2))
            #######expxx#####
            counts += 1
            push!(circuit, HGate(item[1]))
            push!(circuit, HGate(item[2]))
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            push!(circuit, RzGate(item[2], Variable(params[counts])))
            push!(circuit, CNOTGate(control=item[1], target=item[2]))
            push!(circuit, HGate(item[1]))
            push!(circuit, HGate(item[2]))
        end
    end
    return circuit
end
function reparamstoy(param)
    PARAM0=vcat([param[1]],repeat([param[2]], 2))
    PARAM=repeat(PARAM0, length( Edge)*Nlayer)
    return PARAM
end
build_Ansatz(params)=build_toy_Ansatz(reparamstoy(params))
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
function generPsi_x(THETA)
    #100 quantum states of phi are generated from 100 theta parameters
    a=length(THETA[1,:])
    Psi=[]
    circuit=build_Ansatz(THETA[:,1])
    for i in 1:a
        set_parameters!(reparamstoy(THETA[:,i]),circuit)
        x=circuit *ψ
        push!(Psi,x)
    end
    return Psi
end


function loss0(falpha0)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    # alpha=falpha0[1,:].*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    cc=sum(abs.(alpha))
    # deno=real(alpha_Psi'*alpha_Psi)/(cc^2)
    avg_e=real((alpha_Psi'*Ham*alpha_Psi)/(alpha_Psi'*alpha_Psi))                                                                                                                                        
    # println(real(avg_e)," ",deno)              
    # print(f," ", real(avg_e)," ",deno, "\n")             
    return avg_e[1]
end
function loss2(falpha0,De)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    # alpha=falpha0[1,:].*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    cc=sum(abs.(alpha))
    deno=real(alpha_Psi'*alpha_Psi)/(cc^2)
    avg_e1=real((alpha_Psi'*Ham*alpha_Psi)/(alpha_Psi'*alpha_Psi))      
    avg_e2 =((alpha_Psi'*Ham*alpha_Psi)/cc^2)/deno+5*tanh(200*(De-deno))
    # avg_e2=htot*tanh(500*-(deno-0.025))+htot                                                                                                                                    
    println(real(avg_e2)," ",real(avg_e1)," ",deno)              
    print(f," ", real(avg_e2)," ",real(avg_e1)," ",deno, "\n")             
    return real(avg_e2)[1]
end
function loss30(falpha0,De)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    cc=sum(abs.(alpha))
    deno=real(alpha_Psi'*alpha_Psi)/(cc^2)
    return deno
end
function loss3(falpha0,De)
    alpha=exp.(-falpha0[1,:]).*exp.(im*falpha0[2,:])
    alpha_Psi =sum(alpha .* Psi_x)
    cc=sum(abs.(alpha))
    deno=real(alpha_Psi'*alpha_Psi)/(cc^2)
    avg_e1=real((alpha_Psi'*Ham*alpha_Psi)/(alpha_Psi'*alpha_Psi))      
    avg_e2=5*tanh(10*(De-deno))                                                                                                                                 
    println(real(avg_e2)," ",real(avg_e1)," ",deno)              
    # print(f," ", real(avg_e2)," ",real(avg_e1)," ",deno, "\n")             
    return real(avg_e2)[1]
end
function gener_theta()
    #uniformly generate 100 theta parameter vectors
    values_per_partition = 10  # 每个向量包含的值的数量
    step = 2π / values_per_partition  # 步长
    a = [[i * step, j * step] for i in 0:values_per_partition-1, j in 0:values_per_partition-1]
    result = reshape(hcat(a...), (2, 100))
    return result
end
function train_loss_de(t_max,lr_max,lr_min,De)
    i=1
    lr=0.05
    while i<=t_max&&loss30(model(THETA),De)<=De
        opt =ADAM(lr)
        print(i," ")
        # print(f,i," ")
        gs = gradient(() -> loss3(model(THETA),De), ps)
        Flux.Optimise.update!(opt, ps, gs)
        i=i+1
    end
    print(i," ")
    # print(f,i," ")
    loss3(model(THETA),De)
    println("##############")
    # println(f,"##############")
    i=1
    while i<=t_max||(loss0(model(THETA))>=-0.99)
        print(i," ")
        print(f,i," ")
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

end
global ψ=generatepsi(6)
global Nlayer=2
global nedg=length(II)
global Ham=matrix(ham)
global num=2
# f = open("D:\\errormitigation\\toy\\demoninator1\\test2_0_9.txt", "a+")
global Npara=2#The number of variable parameters of a quantum circuit
global model = build_model2( Npara,100,100,2)#neural network model
global THETA=gener_theta()#Generate 100 theta parameter vectors
global Psi_x=generPsi_x(THETA)#100 quantum states of phi are generated from 100 theta parameters
global ps = Flux.params(model)#vectorize the parameters of neural network
#Define the cosin learning rate
global  lr_min=0.00001
global  lr_max=0.0001
global t_max=50000#Defines the maximum number of steps
global De=0.2#Defining the barrier of <1>

# println(f,"model=1,100,100,2 lrmin= ",lr_min,"lr=",lr," lrmax=",lr_max," tmax=",t_max,"de>",De)
train_loss_de(t_max,lr_max,lr_min,De)#Training parameter
close(f)
