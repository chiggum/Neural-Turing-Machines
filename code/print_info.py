import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def print_info(path, ext_in_dim, ext_out_dim, mem_shape, task_name):
    npzfile = np.load(path)
    print "Loaded info from %s." % (path)
    info = npzfile['info'].tolist()
    my_len = info['X'].shape[0]
    info['prediction'] = info['prediction'].reshape((my_len, ext_out_dim))
    info['thresh_prediction'] = info['thresh_prediction'].reshape((my_len, ext_out_dim))
    info['reads'] = info['reads'].reshape((my_len, mem_shape[1]))
    info['adds'] = info['adds'].reshape((my_len, mem_shape[1]))
    info['weights'] = info['weights'].reshape((my_len, mem_shape[0]))
    info['ce_error'] = info['ce_error'].flatten()
    info['hm_error'] = info['hm_error'].flatten()
    
    f, axarr = plt.subplots(5, 2, figsize=(20, 14))
    axarr[0,0].matshow(info['X'].T, aspect='auto')
    axarr[0,0].set_title('X')
    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])
    axarr[0,1].matshow(info['Y'].T, aspect='auto')
    axarr[0,1].set_title('Y')
    axarr[0,1].set_xticks([])
    axarr[0,1].set_yticks([])
    axarr[2,1].matshow(info['prediction'].T, aspect='auto')
    axarr[2,1].set_title('Prediction')
    axarr[2,1].set_xticks([])
    axarr[2,1].set_yticks([])
    axarr[3,1].matshow(np.abs(info['prediction'].T-info['Y'].T), aspect='auto')
    axarr[3,1].set_title('Abs(Prediction - Y)')
    axarr[3,1].set_xticks([])
    axarr[3,1].set_yticks([])
    axarr[1,1].matshow(info['thresh_prediction'].T, aspect='auto')
    axarr[1,1].set_title('Thresholded Predition')
    axarr[1,1].set_xticks([])
    axarr[1,1].set_yticks([])
    axarr[2,0].matshow(info['reads'].T, aspect='auto')
    axarr[2,0].set_title('Reads')
    axarr[2,0].set_xticks([])
    axarr[2,0].set_yticks([])
    axarr[1,0].matshow(info['adds'].T, aspect='auto')
    axarr[1,0].set_title('Adds')
    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])
    axarr[3,0].matshow(info['weights'][:my_len/2+1,:], aspect='auto', cmap = cm.Greys_r)
    axarr[3,0].set_title('Weights till second delimeter')
    axarr[3,0].set_xticks([])
    axarr[3,0].set_yticks([])
    axarr[4,0].matshow(info['weights'][(my_len/2+1):,:], aspect='auto', cmap = cm.Greys_r)
    axarr[4,0].set_title('Weights after second delimeter')
    axarr[4,0].set_xticks([])
    axarr[4,0].set_yticks([])
    axarr[4,1].axis([0, 10, 0, 10])
    axarr[4,1].set_title(task_name)
    axarr[4,1].set_xticks([])
    axarr[4,1].set_yticks([])
    axarr[4,1].text(2, 6, ('Sequence Length: ' + str((my_len-2)/2) + '\nCE-Error: ' + str(info['ce_error'][0])
                            + '\nHamming Error: ' + str(info['hm_error'][0])) , fontsize=10)
    plt.axis('off')
    plt.setp(axarr[0,0].get_xticklabels(), visible=False)
    plt.setp(axarr[0,0].get_yticklabels(), visible=False)
    plt.setp(axarr[0,1].get_xticklabels(), visible=False)
    plt.setp(axarr[0,1].get_yticklabels(), visible=False)
    plt.setp(axarr[1,0].get_xticklabels(), visible=False)
    plt.setp(axarr[1,0].get_yticklabels(), visible=False)
    plt.setp(axarr[1,1].get_xticklabels(), visible=False)
    plt.setp(axarr[1,1].get_yticklabels(), visible=False)
    plt.setp(axarr[2,0].get_xticklabels(), visible=False)
    plt.setp(axarr[2,0].get_yticklabels(), visible=False)
    plt.setp(axarr[2,1].get_xticklabels(), visible=False)
    plt.setp(axarr[2,1].get_yticklabels(), visible=False)
    plt.setp(axarr[3,0].get_xticklabels(), visible=False)
    plt.setp(axarr[3,0].get_yticklabels(), visible=False)
    plt.setp(axarr[3,1].get_xticklabels(), visible=False)
    plt.setp(axarr[3,1].get_yticklabels(), visible=False)
    plt.tight_layout()
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    plt.savefig(path.split('.npz')[0]+'.png')
    plt.clf()

def print_learning_curve(path, task_name, step_size=10):
    X = np.genfromtxt(path, delimiter=',')
    mylen = X.shape[0]
    print X[-1,:]
    ind = np.arange(0, mylen, step_size)
    fig = plt.figure()
    plt.plot(X[ind,0], X[ind,3], 'ro-')
    plt.title(('Learning Curve for ' + task_name))
    plt.xlabel('Iteration no.')
    plt.ylabel('Hamming Distance')
    plt.savefig(path.split('.csv')[0]+'.png')
    plt.clf()