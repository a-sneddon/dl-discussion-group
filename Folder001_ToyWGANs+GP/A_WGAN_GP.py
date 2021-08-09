###===>>>+++++++++++++++++++++++
# From Nic, to friends in ANU. +
###===###+++++++++++++++++++++++

###===>>>
# (Part 0) Some helpful material
# 	Step I.   -- For understanding GANs
# 		https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29
# 	Step II.  -- For understanding a simple implementation of GANs
# 		https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
# 	Step III. -- For understanding WGANs
# 		https://arxiv.org/pdf/1701.07875.pdf
# 	Step IV.  -- For understanding gradient penalty
# 		https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
#
# Note, this line of material is a bit more mathematical,
# but most of them are beautifully written and made simple.
# This includes the material in step III. -- the original WGANs paper --
# the author took a really casual writing style and explained everything every nicely. :3

###===>>>
# (Part 1) Load dependencies

#---
# torch related
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim

#---
# common packages
import  pandas                  as      pd
import  numpy                   as      np
import  copy
import  random
import  matplotlib.pyplot       as      plt

#---
# my own packages
from    B_MyModels              import  *
from    C_PlotResults           import  *

###===>>>
# (Part 2)      Seed our system for reproducibility
seed = 0
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed(     seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###===>>>
# (Part 3)      Set misc.

#---
# This is where we will store our stuff
SAVE_PATH = './D_Folder/'

#---
# Succesfully training a GANs network is time costly
# so we allow the option to continue train from our last saved model
Cont_Train = False

###===>>>
# (Part 4)      Hyperparameters
#---
# Data setup
Data001_Seq     = 2         # We will generating 2 sine waves
Data002_Hieght  = 0.995     # bounded in [0, 0.995]
Data003_Len     = 10        # with length 10
Data004_Prop    = 0.25      # and propagates 0.25 pi per increment

# The first half of the toy data is here
X_Prog = torch.tensor(
            np.array([i * Data004_Prop for i in range(Data003_Len)])).float()
X_Prog = X_Prog.view(-1, Data003_Len, 1)

# The next half is in the training part

#---
# Experiment setup
Exp001_MaxIter      = 1000  # We will train for 1000 iterations
Exp002_BatchSize    = 128   # each of batch size 128
Exp003_CriticIters  = 5     # and train the generator once every 5 times
                            # of that of the discriminator

#---
# Network setup
Net001_ID   = 16            # We will sample from 16 latent input dimensions (ID)
Net002_HD   = 128           # and use hidden dimension (HD) of 128

#---
# Adam Optimiser setup
Opt001_LR       = 1e-3          # We will use learning rates of 1e-3
Opt002_Betas    = (0.9, 0.999)  # and use the standard Adam moments 

###===>>>
# (Part 5)      Start new or continue

#---
if not Cont_Train:
    # if we are starting from a clean canvas
    AlreadyTrained = 0
    NewIterations = Exp001_MaxIter

else:
    # if continuing 
    # we check how much more new iterations are we aiming for
    AlreadyTrained  = torch.load(SAVE_PATH + 'AlreadyTrained')
    NewIterations   = Exp001_MaxIter - AlreadyTrained

    if NewIterations <= 0:
        raise AssertionError("Cont_Trained Exp001_MaxIter <= 0")
        
#---
if not Cont_Train:
    # if we are starting from a clean canvas
    # define the generator
    G_Network = Generator(
                    Net001_ID, Net002_HD,
                    Data001_Seq)

    # define the discriminator
    D_Network = Discriminator(
                    Net002_HD,
                    Data001_Seq, Data003_Len)
    
    G_Network = G_Network.cuda()
    D_Network = D_Network.cuda()

    #---
    # define some storage for matplotlib targets
    DLoss_all = []
    GLoss_all = []
    PLoss_all = []

else:
    # if continue,
    # load the old data
    G_Network   = torch.load(SAVE_PATH + 'G_Network')
    D_Network   = torch.load(SAVE_PATH + 'D_Network')

    #---
    DLoss_all = torch.load(SAVE_PATH + 'DLoss_all')
    GLoss_all = torch.load(SAVE_PATH + 'GLoss_all')
    PLoss_all = torch.load(SAVE_PATH + 'PLoss_all')

#---
# Note here that no matter continue or not,
# we use new Adam optimisers
# this is because loading Adam state_dict is tricky
G_optimiser = optim.Adam(G_Network.parameters(), lr = Opt001_LR,
                         betas = Opt002_Betas)
D_optimiser = optim.Adam(D_Network.parameters(), lr = Opt001_LR,
                         betas = Opt002_Betas)

###===>>>
# (Part 6)      Training
# For each iteration
for CurIteration in range(NewIterations):

    print("###===###")
    print("Iter: {}".format(CurIteration + 1 + AlreadyTrained))

    #---
    # Creating some training dataset

    # start with random beginnings
    My_X   = torch.randn(Exp002_BatchSize, 1, 1).float()

    # add the propagation
    My_X   = My_X + X_Prog

    # and define the waves
    My_X1   = (Data002_Hieght * torch.sin(My_X * np.pi) + 1)/2
    My_X2   = (Data002_Hieght * torch.sin(My_X * np.pi + np.pi/2) + 1)/2

    My_X = torch.cat((My_X1, My_X2), dim = 2)

    Real_X = My_X.cuda()

    #---
    # Update the discriminator & generator at a 5-1 ratio
    for CurCriticIteration in range(Exp003_CriticIters):

        #---
        # sample the latent inputs for the generator
        z = torch.rand(Exp002_BatchSize, Data003_Len, Net001_ID).cuda()
        # and create the fake data
        Fake_X = G_Network(z)

        #---
        # rate the waves
        D_real = D_Network(Real_X)
        D_fake = D_Network(Fake_X)

        # and enforce smoothness via gradient penalty
        GP_reg = D_Network.GP(Real_X, Fake_X, Exp002_BatchSize)

        #---
        # Update the discriminator base on
        # "how uncertain it is on the realness of the fake data"
        D_optimiser.zero_grad()
        D_loss = D_fake.mean() - D_real.mean() + GP_reg
        D_loss.backward()
        D_optimiser.step()

        #---
        # and record the historic values
        DLoss_all.append(D_loss.item())
        PLoss_all.append(GP_reg.item())

    #---
    # To update the generator
    # sample the latent variables
    z = torch.rand(Exp002_BatchSize, Data003_Len, Net001_ID).cuda()
    # create the fake waves
    Fake_X = G_Network(z)
    # and rate the fake waves
    D_fake = D_Network(Fake_X)

    # update the generator base on
    # "how less fake it could be"
    G_optimiser.zero_grad()
    G_loss = -D_fake.mean()
    G_loss.backward()
    G_optimiser.step()

    GLoss_all.append(G_loss.item())

    #---
    # print the running results
    print("---------")
    print("DLoss: \t{}".format(D_loss.item()))
    print("PLoss: \t{}".format(GP_reg.item()))
    print("GLoss: \t{}".format(G_loss.item()))
    print("")

###===>>>
# (Part 7)      Save the results and plotting them

#---
# The behaviours of the loss functions
plt.plot([i for i in range(1, len(DLoss_all)+1)], DLoss_all)
plt.plot([i for i in range(1, len(DLoss_all)+1)], PLoss_all)

# have to be careful when we plot the generator loss
# because it was updated at a different pace
plt.plot([i for i in range(Exp003_CriticIters, len(DLoss_all)+Exp003_CriticIters, Exp003_CriticIters)],
         GLoss_all,
         'o-')

plt.legend(['Discriminator Loss',
            'GP Regularisation',
            'Generator Loss'])

plt.savefig(SAVE_PATH + 'A001_LossComparison_Iters{}.png'.format(Exp001_MaxIter))

#---
# and next we generate some results 
PlotResults(Fake_X, Real_X, Data003_Len, SAVE_PATH, Exp001_MaxIter)

#---
# and conclude by saving the models 
torch.save(G_Network,                   SAVE_PATH + 'G_Network')
torch.save(D_Network,                   SAVE_PATH + 'D_Network')

torch.save(DLoss_all,                   SAVE_PATH + 'DLoss_all')
torch.save(GLoss_all,                   SAVE_PATH + 'GLoss_all')
torch.save(PLoss_all,                   SAVE_PATH + 'PLoss_all')

torch.save(Exp001_MaxIter,              SAVE_PATH + 'AlreadyTrained')

         
