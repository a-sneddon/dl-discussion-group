###===###
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim
from    torch                   import  autograd

from torch.nn.utils import spectral_norm

###===###
import  pandas                  as      pd
import  numpy                   as      np
import  copy
import  random
import  matplotlib.pyplot       as      plt

###===>>>
# Below, we painfully define an LSTM because
# my outdated PyTorch cannot handle RNN backward computations
# in gradient penalty calculation (see GP in the discriminator)
class MyLSTM(nn.Module):
    def __init__(self, ID, HD):
        super().__init__()
        self.ID = ID
        self.HD = HD
        
        self.i2h = nn.Linear(ID, HD * 4)
        self.h2h = nn.Linear(HD, HD * 4)

    def forward(self, x0):

        Q_k = torch.zeros(x0.shape[0], self.HD).cuda()
        S_k = torch.zeros(x0.shape[0], self.HD).cuda()

        Q_all = []

        for QStr in range(x0.shape[1]):

            X_k = x0[:, QStr, :]
            
            #---
            F_i, I_i, A_i, O_i = self.i2h(X_k).chunk(4, dim = 1)
            F_h, I_h, A_h, O_h = self.h2h(Q_k).chunk(4, dim = 1)

            F_k = torch.sigmoid(F_i + F_h)
            I_k = torch.sigmoid(I_i + I_h)
            A_k = torch.tanh(   A_i + A_h)
            O_k = torch.sigmoid(O_i + O_h)                

            ###===###
            S_k = F_k * S_k + I_k * A_k
            Q_k = O_k * torch.tanh(S_k)

            Q_all.append(Q_k.unsqueeze(1))

        Q_all = torch.cat(Q_all, dim = 1)

        return Q_k, Q_all

###===>>>
class Generator(nn.Module):

    ###===###
    def __init__(self,
                 Net001_ID, Net002_HD,
                 Data001_Seq):
        super().__init__()

        #---
        ID = Net001_ID
        HD = Net002_HD
        OD = Data001_Seq # the output dimension is the total amount of sequences

        #---
        # After sampling from the latent dimension
        # we will pass through a bidirectional LSTM
        # f for forward, and r for reverse
        self.RNNf = MyLSTM(ID, HD)
        self.RNNr = MyLSTM(ID, HD)

        #---
        # and then feed infer through 3 layers
        self.L1 = nn.Linear(2 * HD, HD)
        self.L2 = nn.Linear(HD, HD)
        self.L3 = nn.Linear(HD, OD)

        # of which are activated with the LeakyReLU function
        self.LRU = nn.LeakyReLU(0.2)

    ###===###
    def forward(self, x0):

        #---
        # x0f is the input of the forward LSTM
        x0f = x0
        # and we flip the sequence for the backward LSTM
        x0r = x0.flip(dims = [1])

        #---
        _, RNNf_all = self.RNNf(x0f)
        _, RNNr_all = self.RNNr(x0r)
        x1 = torch.cat((RNNf_all, RNNr_all), dim = 2)

        #---
        x2 = self.LRU( self.L1(x1))
        x3 = self.LRU( self.L2(x2))

        out = torch.sigmoid(self.L3(x3))

        return out

###===>>>
class Discriminator(nn.Module):

    ###===###
    def __init__(self,
                 Net002_HD,
                 Data001_Seq, Data003_Len):
        super().__init__()

        #---
        ID = Data001_Seq
        HD = Net002_HD

        #---
        # we will first pass the sequences through dense layers
        self.L1 = nn.Linear(ID, HD)
        self.L2 = nn.Linear(HD, HD)

        #---
        # and then evaluate them with convolutional layers
        self.C1 = nn.Conv1d(HD, 32, 3, padding = 1)
        self.C2 = nn.Conv1d(32,  8, 3, padding = 1)
        self.C3 = nn.Conv1d( 8,  2, 3, padding = 1)

        #---
        # and finally rate how real the data is
        self.L3 = nn.Linear(2 * Data003_Len, 1)

        # again, the intermediate layers are activated with LeakyReLU
        self.LRU = nn.LeakyReLU(0.2)

    ###===###
    def forward(self, x0):

        #---
        x1 = self.LRU( self.L1(x0))
        x2 = self.LRU( self.L2(x1))

        #---
        x2 = x2.transpose(1, 2)
        x2 = torch.relu(self.C1(x2))
        x2 = torch.relu(self.C2(x2))
        x2 = torch.relu(self.C3(x2))

        x3 = x2.view(x2.shape[0], -1)

        #---
        out = self.L3(x3)

        return out

    ###===###
    # the variant of WGAN we are looking at is
    # the gradient penalty variant
    # this is for stabilising the training procedure
    # and ensuring the Lipschitz condition
    def GP(self,
           Real_X, Fake_X,
           Exp002_BatchSize
           ):

        GP_Lambda = 10

        ###===###
        # Gradient penalty enforces smoothness to WGAN by
        # measuring the deviation of extrapolated quasi-real data on the critic
 	# from the ideal normalised gradient of 1

        # we first define the mixing constant
        alpha = torch.rand(Exp002_BatchSize, 1, 1).cuda()
        alpha = alpha.expand_as(Real_X)

        # and we extrapolate the data
        Re_X = alpha * Real_X + (1 - alpha) * Fake_X

        # pass it through the discriminator
        prob_interpolated = self(Re_X)

        # and compute how far away from real it is
        # so the gradient output targets is from a vector of ones
        gradients = autograd.grad(
                        outputs         = prob_interpolated,
                        inputs          = Re_X,
                        grad_outputs    =\
                            torch.ones_like(prob_interpolated).cuda(),
                        create_graph    = True,
                        retain_graph    = True,
                    )[0]

        # and then we flatten the gradient
        gradients = gradients.contiguous().view(Exp002_BatchSize, -1)

        # in order to take the norm
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # and apply the regularisation weight
        GP_Lambda = 10
        GP_reg = GP_Lambda * ((gradients_norm - 1) ** 2).mean()

        return GP_reg
