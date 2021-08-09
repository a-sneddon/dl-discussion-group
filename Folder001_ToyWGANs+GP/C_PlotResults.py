###===###
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim

###===###
import  pandas                  as      pd
import  numpy                   as      np
import  copy
import  random
import  matplotlib.pyplot       as      plt

def PlotResults(Fake_X, My_Target,
                SequenceLen,
                SAVE_PATH, MaxIterations):
    fig = plt.figure(figsize = (8, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.subplot(2, 2, 1)
    plt.plot(Fake_X[1, :, :2].cpu().detach().numpy())
    plt.title('Sample 1')
    plt.xticks([])
    plt.yticks(fontsize=10)
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 2)
    plt.plot(Fake_X[2, :, :2].cpu().detach().numpy())
    plt.title('Sample 2')
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 3)
    plt.plot(Fake_X[3, :, :2].cpu().detach().numpy())
    plt.title('Sample 3')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 4)
    plt.plot(Fake_X[4, :, :2].cpu().detach().numpy())
    plt.title('Sample 4')
    plt.xticks(fontsize=10)
    plt.yticks([])
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')

    fig.suptitle('Generated Data')
    plt.savefig(SAVE_PATH + 'A002_GeneratedData_Iters{}.png'.\
                format(MaxIterations))

    fig = plt.figure(figsize = (8, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.subplot(2, 2, 1)
    plt.plot(My_Target[1, :, :2].cpu().detach().numpy())
    plt.title('Sample 1')
    plt.xticks([])
    plt.yticks(fontsize=10)
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 2)
    plt.plot(My_Target[2, :, :2].cpu().detach().numpy())
    plt.title('Sample 2')
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 3)
    plt.plot(My_Target[3, :, :2].cpu().detach().numpy())
    plt.title('Sample 3')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    
    plt.subplot(2, 2, 4)
    plt.plot(My_Target[4, :, :2].cpu().detach().numpy())
    plt.title('Sample 4')
    plt.xticks(fontsize=10)
    plt.yticks([])
    plt.ylim([-0.125, 1.125])
    plt.hlines( 0.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.25, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.50, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 0.75, 0, SequenceLen, linestyles = 'dashed', colors = 'k')
    plt.hlines( 1.00, 0, SequenceLen, linestyles = 'dashed', colors = 'k')

    fig.suptitle('Real Data')
    plt.savefig(SAVE_PATH + 'A003_RealData_Iters{}.png'.\
                format(MaxIterations))





    
