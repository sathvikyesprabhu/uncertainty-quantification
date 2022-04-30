import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

import scipy.ndimage as ndim
import matplotlib.colors as mcolors
conv = mcolors.ColorConverter().to_rgb
# textsize = 15
# marker = 5

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  

def plot_cost(pred_cost_train,cost_dev,nb_epochs,results_dir):
    plt.figure(dpi=100)
    plt.plot(range(0, nb_epochs), cost_dev, 'b-')
    plt.plot(range(0, nb_epochs),pred_cost_train, 'r--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train loss', 'Test loss'])
    plt.title('Classification Loss')
    plt.savefig(results_dir + '/pred_cost.png', bbox_inches='tight')
    plt.show()
    
def plot_error(err_train, err_dev, nb_epochs, results_dir):
    plt.figure(dpi=100)
    plt.semilogy(range(0, nb_epochs), err_dev, 'b-')
    plt.semilogy(100 * err_train, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid()
    plt.legend(['Test error', 'Train error'])
    plt.savefig(results_dir + '/err.png', box_inches='tight')
    plt.show()

def plot_KL_cost(kl_cost_train,results_dir):
    plt.figure(dpi=100)
    plt.plot(kl_cost_train, 'r')
    plt.ylabel('Nats')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train error', 'Test error'])
    plt.title('DKL (per sample)')
    plt.savefig(results_dir + '/KL_cost.png', bbox_inches='tight')
    plt.show()

def plot_rotate(x_dev, y_dev, net, results_dir, im_list, im_ind = 90,Nsamples = 100, steps = 10):
    ## ROTATIONS marginloss percentile distance

    plt.figure()
    # plt.imshow( ndim.interpolation.rotate(np.transpose(x_dev[im_ind,:,:,:],(1,2,0)), 0, reshape=False))
    plt.imshow(im_list[im_ind])
    plt.title('original image')
    plt.savefig(results_dir + '/sample_image.png', bbox_inches='tight')
    s_rot = 0
    end_rot = 179
    rotations = (np.linspace(s_rot, end_rot, steps)).astype(int)            

    ims = []
    predictions = []
    # percentile_dist_confidence = []
    x, y = x_dev[im_ind], y_dev[im_ind]

    fig = plt.figure(figsize=(steps, 8), dpi=80)

    # DO ROTATIONS ON OUR IMAGE

    for i in range(len(rotations)):
        
        angle = rotations[i]
        x_rot = ndim.interpolation.rotate(x, angle, axes=(1,2),reshape=False, mode='nearest')
        
        ax = fig.add_subplot(3, (steps-1), 2*(steps-1)+i)  
        # ax.imshow(np.transpose(x_rot,(1,2,0))) # Image pixels lie in [-1,1]
        ax.imshow(ndim.interpolation.rotate(im_list[im_ind],angle, axes=(0,1),reshape=False ,mode='nearest'))
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ims.append(x_rot)
        
    ims = np.array(ims)
    print(ims.shape)
    y = np.ones(ims.shape[0])*y
    # ims = np.expand_dims(ims, axis=1)

    with torch.no_grad():
        cost, err, probs = net.sample_eval(torch.from_numpy(ims), torch.from_numpy(y), Nsamples=Nsamples, logits=False) # , logits=True

    predictions = probs.numpy()    
    textsize = 15
    lw = 5

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  

    ax0 = plt.subplot2grid((3, steps-1), (0, 0), rowspan=2, colspan=steps-1)
    #ax0 = fig.add_subplot(2, 1, 1)
    plt.gca().set_prop_cycle(color = c)
    ax0.plot(rotations, predictions, linewidth=lw)


    ##########################
    # Dots at max

    for i in range(predictions.shape[1]):

        selections = (predictions[:,i] == predictions.max(axis=1))
        for n in range(len(selections)):
            if selections[n]:
                ax0.plot(rotations[n], predictions[n, i], 'o', c=c[i], markersize=15.0)
    ##########################  

    lgd = ax0.legend(['airplane', 'automobile', 'bird',
                'cat', 'deer', 'dog',
                'frog', 'horse', 'ship',
                'truck'], loc='upper right', prop={'size': textsize, 'weight': 'normal'}, bbox_to_anchor=(1.35,1))
    plt.xlabel('rotation angle')
    # plt.ylabel('probability')
    plt.title('True class: %d, Nsamples %d' % (y[0], Nsamples))
    # ax0.axis('tight')
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    for item in ([ax0.title, ax0.xaxis.label, ax0.yaxis.label] +
                ax0.get_xticklabels() + ax0.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')

    plt.savefig(results_dir + '/percentile_label_probabilities.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # files.download('percentile_label_probabilities.png')

def get_predictions(x_dev, y_dev, net, Nsamples = 100, steps = 16):
    # All samples + entropy
    # (Nsamples) samples, for (steps) rotation steps, for 10,000 val images
    tic = time.time()
    s_rot = 0
    end_rot = 179
    rotations = (np.linspace(s_rot, end_rot, steps)).astype(int)            
    
    all_preds = np.zeros((len(x_dev), steps, 10))
    all_sample_preds = np.zeros((len(x_dev), Nsamples, steps, 10))

    # DO ROTATIONS ON OUR IMAGE
    for im_ind in range(len(x_dev)):
        if(im_ind % 500 == 0):
            print(im_ind)
        x, y = x_dev[im_ind], y_dev[im_ind]
        
        ims = []
        predictions = []
        for i in range(len(rotations)):

            angle = rotations[i]
            x_rot = ndim.interpolation.rotate(x, angle, axes=(1,2), reshape=False)
            ims.append(x_rot)

        ims = np.array(ims)
        
        y = np.ones(ims.shape[0])*y
        
    #     cost, err, probs = net.sample_eval(torch.from_numpy(ims), torch.from_numpy(y), Nsamples=Nsamples, logits=False)
        with torch.no_grad():
            sample_probs = net.all_sample_eval(torch.from_numpy(ims), torch.from_numpy(y), Nsamples=Nsamples)
        probs = sample_probs.mean(dim=0)
        
        all_sample_preds[im_ind, :, :, :] = sample_probs.cpu().numpy()
        predictions = probs.cpu().numpy()
        all_preds[im_ind, :, :] = predictions
    
    
    correct_preds = np.zeros((len(x_dev), steps))
    for i in range(len(x_dev)):
        correct_preds[i,:] = all_preds[i,:,y_dev[i]]

    toc = time.time()
    print("Time taken: ", toc-tic, " s")
    return correct_preds, all_preds, all_sample_preds, rotations

def plot_predictive_entropy(correct_preds, all_preds, rotations, results_dir):
    all_preds_entropy = -(all_preds * np.log(all_preds)).sum(axis=2)
    mean_angle_entropy = all_preds_entropy.mean(axis=0)
    std_angle_entropy = all_preds_entropy.std(axis=0)

    correct_mean = correct_preds.mean(axis=0)
    correct_std = correct_preds.std(axis=0)

    plt.figure(dpi=100)
    line_ax0 = errorfill(rotations, correct_mean, yerr=correct_std, color=c[2])
    ax = plt.gca()
    ax2 = ax.twinx()
    line_ax1 = errorfill(rotations, mean_angle_entropy, yerr=std_angle_entropy, color=c[3], ax=ax2)
    plt.xlabel('rotation angle')
    lns = line_ax0+line_ax1

    lgd = plt.legend(lns, ['correct class', 'predictive entropy'], loc='upper right',
                    prop={'size': 15, 'weight': 'normal'}, bbox_to_anchor=(1.75,1))


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + [ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(15)
        item.set_weight('normal')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(results_dir + '/predictive_entropy.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    line_ax = ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    
    return line_ax
