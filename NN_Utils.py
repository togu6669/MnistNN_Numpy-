import numpy as np
import NeuronLayers2


def SaveTheNet (HiddenLayer1, HiddenLayer2, OutputLayer, epoch, test_img = None):
    
    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    HiddenLayer1.layerinputs.reshape(-1, 1)
    HiddenLayer2.layerinputs.reshape(-1, 1)
    OutputLayer.layerinputs.reshape(-1, 1)

    if test_img == None:
        filename = "MNIST-inputs-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-inputs-" + str(epoch) + "-image-" + str(test_img) +".txt"

    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layerinputs.shape [0]):
        file2write.write (str(HiddenLayer1.layerinputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 2-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer2.layerinputs.shape [0]):
        file2write.write (str(HiddenLayer2.layerinputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layerinputs.shape [0]):
        file2write.write (str(OutputLayer.layerinputs [a])+ '\n')

    file2write.close()

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    if test_img == None:
        filename = "MNIST-weights-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-weights-" + str(epoch) + "-image-" + str(test_img) +".txt"
    
    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layerweights.shape [0]):
        file2write.write (str(HiddenLayer1.layerweights [a]))

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 2-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer2.layerweights.shape [0]):
        file2write.write (str(HiddenLayer2.layerweights [a]))

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layerweights.shape [0]):
        file2write.write (str(OutputLayer.layerweights [a]))

    file2write.close()

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    HiddenLayer1.layeroutputs.reshape(-1, 1)
    HiddenLayer2.layeroutputs.reshape(-1, 1)
    OutputLayer.layeroutputs.reshape(-1, 1)

    if test_img == None:
        filename = "MNIST-outputs-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-outputs-" + str(epoch) + "-image-" + str(test_img) +".txt"

    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layeroutputs.shape [0]):
        file2write.write (str(HiddenLayer1.layeroutputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 2-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer2.layeroutputs.shape [0]):
        file2write.write (str(HiddenLayer2.layeroutputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layeroutputs.shape [0]):
        file2write.write (str(OutputLayer.layeroutputs [a]) + '\n')

    file2write.close()

    
def SaveTheNet1Hidden (HiddenLayer1, OutputLayer, epoch, test_img = None):

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    HiddenLayer1.layerinputs.reshape(-1, 1)
    OutputLayer.layerinputs.reshape(-1, 1)

    if test_img == None:
        filename = "MNIST-inputs-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-inputs-" + str(epoch) + "-image-" + str(test_img) +".txt"

    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layerinputs.shape [0]):
        file2write.write (str(HiddenLayer1.layerinputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layerinputs.shape [0]):
        file2write.write (str(OutputLayer.layerinputs [a])+ '\n')

    file2write.close()

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    if test_img == None:
        filename = "MNIST-weights-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-weights-" + str(epoch) + "-image-" + str(test_img) +".txt"
    
    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layerweights.shape [0]):
        file2write.write (str(HiddenLayer1.layerweights [a]))

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layerweights.shape [0]):
        file2write.write (str(OutputLayer.layerweights [a]))

    file2write.close()

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------

    HiddenLayer1.layeroutputs.reshape(-1, 1)
    OutputLayer.layeroutputs.reshape(-1, 1)

    if test_img == None:
        filename = "MNIST-outputs-" + str(epoch) + ".txt"
    else:
        filename = "MNIST-outputs-" + str(epoch) + "-image-" + str(test_img) +".txt"

    file2write = open (filename,'w')
    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("--------------------------------HIDDEN LAYER 1-------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (HiddenLayer1.layeroutputs.shape [0]):
        file2write.write (str(HiddenLayer1.layeroutputs [a])+ '\n')

    file2write.write ("-----------------------------------------------------------------------------")
    file2write.write ("-------------------------------- OUTPUT LAYER -------------------------------")
    file2write.write ("-----------------------------------------------------------------------------")
    for a in range (OutputLayer.layeroutputs.shape [0]):
        file2write.write (str(OutputLayer.layeroutputs [a]) + '\n')

    file2write.close()
