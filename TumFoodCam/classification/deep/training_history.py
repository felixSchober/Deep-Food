import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from misc import utils
import data_io.settings as Settings
from nolearn.lasagne.visualize import plot_conv_weights, plot_conv_activity, plot_occlusion, occlusion_heatmap

def plot_train_history(trainHistory, netId, netName, elements, yLabel, title, ylimit=(1e-2, 1e-0), yscale="log", xLimit=(0, Settings.NN_EPOCHS), path=None, fileName=None):
    """ 
    Function that is called after epochs are finished to plot the current training state. 
    
    Keyword arguments:
    trainHistory -- nolearn type training history
    netId -- net ID for the diagram.
    netName -- name of the net.
    elements -- index of the nolearn training history elements and labels for the diagram. (e.g. [("train_loss", "train loss"), ("valid_loss", "valid loss")])
    yLabel -- label for the y-axis
    ylimit -- limits of the y-axis
    yscale -- scale of the y-axis
    xLimit -- limits of the x-axis
    path -- save path for saving
    fileName -- file name for saving
    """
    
    plt.clf()
    plt.figure(figsize=(12,9), dpi=100)

    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)      
    ax.spines["right"].set_visible(False)    
  
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    


    colorIndex = 0
    for hist in elements:
        historyKey = hist[0]
        historyLabel = hist[1]
        e = np.array([i[historyKey] for i in trainHistory])
        plt.plot(e, linewidth=3, label=historyLabel, color=Settings.G_COLOR_PALETTE[colorIndex]) 
        colorIndex += 1               
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(yLabel)
    if ylimit != None:
        plt.ylim(ylimit[0], ylimit[1])
    if yscale != None:
        plt.yscale(yscale)
    plt.xlim(xLimit[0], xLimit[1])
    plt.title("{0} training results for NN {1}".format(title, netName), fontsize=22)  
    minY, maxY = ax.get_ylim()
    ypos = minY + 0.02*(maxY - minY) # position slightly above x-axis
    plt.text(20, ypos, str(netId)) # net id at the top left      
    
    if fileName is None:
        fileName = netName + "_trainingResults_" + title
    return utils.save_plt_figure(plt, fileName, path)

# for video animation of history run
#       ffmpeg -framerate 25 -i %01d.png -vcodec libx264 -r 30 -pix_fmt yuv420p out.mp4


class TrainingHistory(object):
    """
    This class gets called after every nn epoch and plots certain parameters of the net for later visualization.
    1. current train_loss and accuracy
    2. current conv. activations if cnn
    """

    def __init__(self, name, netId, plotLayerWeigths, modelSaver):
        self.name = name
        self.netId = netId
        self.plotLayers = plotLayerWeigths

        path = modelSaver.get_save_path_for_visualizations() + "train_history/"
        self.lossPath = path + "loss/"
        self.accPath = path + "accuracy/"

        utils.create_dir_if_necessary(path)
        utils.create_dir_if_necessary(self.lossPath)
        utils.create_dir_if_necessary(self.accPath)

        if plotLayerWeigths:
            # create dir for each layer
            self.cWeightsPath = path + "conv_weights/"        
            utils.create_dir_if_necessary(self.cWeightsPath)
            for layer in plotLayerWeigths:                      
                utils.create_dir_if_necessary(self.cWeightsPath + str(layer) + "/")

    def __call__(self, nn, train_history): 
        epoch = train_history[-1]['epoch']

        # plot current loss and accuracy
        plot_train_history(train_history, self.netId, self.name, [("train_loss", "train loss"), ("valid_loss", "valid loss")], "loss", "Loss", ylimit=None, yscale=None, xLimit=(0, Settings.NN_EPOCHS), path=self.lossPath, fileName=str(epoch))
        plot_train_history(train_history, self.netId, self.name, [("valid_accuracy", "Accuracy")], "accuracy", "Accuracy", ylimit=None, yscale=None, xLimit=(0, Settings.NN_EPOCHS), path=self.accPath, fileName=str(epoch))

        if self.plotLayers:
            for layer in self.plotLayers:                
                plt = plot_conv_weights(nn.layers_[layer], figsize=(6,6))
                utils.save_plt_figure(plt, str(epoch), self.cWeightsPath + str(layer) + "/")
                plt.close("all")



