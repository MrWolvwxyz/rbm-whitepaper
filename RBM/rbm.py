import os, struct, sys
import image
from scipy import signal
from array import array as pyarray
from time import sleep
import matplotlib as plt
from pylab import *
from numpy import *

def logistic( x ):
    return 1.0 / ( 1.0 + exp( -x ) )

def ff( x ):
    return flipud( fliplr( x ) )
    
def init_activation( x ):
    #For teh dropout
    #mask = 0.75 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( x )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    
def init_activation_conv( x ):
    #For teh dropout
    #mask = 0.75 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( x )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ], x.shape[ 2 ], x.shape[ 3 ] )

def init_activation_conv_recon( x ):
    #For teh dropout
    #mask = 0.75 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( x )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ], x.shape[ 2 ] )

def drop_activation( x ):
    #For teh dropout
    mask = 0.5 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( multiply( x, mask ) )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ] )

def load_mnist( dataset = "training", digits = arange( 10 ),
                path = "/Users/Sam/Documents/rbm-whitepaper/RBM" ):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    images_conv = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows * cols))
        images_conv[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels, images_conv

       
num_visible = 784
num_hidden = 1000
batch = 6000
set_size = 60000
size = 28
num_case = set_size / batch
rate = 0.0025
pbias_lambda = 5
l2 = 0.01
epoch = 10
max_epoch_1 = 10
max_epoch_2 = 10
init_mom = 0.5
final_mom = 0.9

images, labels, images_conv = load_mnist( 'training' )
tr_images = images / 255.0

weights = 0.001 * random.randn( num_visible, num_hidden )
hid_bias = zeros( num_hidden )
vis_bias = zeros( num_visible )

w_grad = zeros( ( num_visible, num_hidden ) )
hb_grad = zeros( num_hidden )
vb_grad = zeros( num_visible )

for i in range( num_visible ):
    if vis_bias[ i ] == -inf:
        vis_bias[ i ] = 0
 
print( "training standard rbm" )
if not max_epoch_1: print( "haha not, crbm cooler" )       
for i in range(10):
   for tr_batch in range( set_size // batch ):
        print( "epoch ", i, " batch ", tr_batch ) 
        momentum = 0
        
        #positive phase
        data = tr_images[ batch * tr_batch : batch * ( 1 + tr_batch ), : ] > rand( batch, num_visible )
        hid_prob, hid_act = init_activation( hid_bias + 2 * dot( data, weights ) )
        
        #negative phase
        vis_prob, vis_act = init_activation( vis_bias + dot( hid_act, transpose( weights ) ) )
        hid_rec_prob, hid_rec_act = init_activation( hid_bias + 2 * dot( vis_act, weights ) )
            
        #update momentum
        if i > 5:
            momentum = final_mom
        else:
            momentum = init_mom
        
        #compute gradients and add them to the parameters
        w_grad = w_grad * momentum + rate * ( ( dot( transpose( data ), hid_prob )
                          - dot( transpose( vis_act ), hid_rec_prob ) ) / num_case - l2 * weights - pbias_lambda * w_grad )
        vb_grad = momentum * vb_grad + rate * ( data - vis_act ).mean( axis = 0 )
        hb_grad = momentum * hb_grad + rate * ( hid_prob - hid_rec_prob ).mean( axis = 0 )
        
        #update parameters
        weights += w_grad
        vis_bias += vb_grad
        hid_bias += hb_grad
        
        
        #display example/reconstructions
        if(epoch == 9):
            fig = figure()
            for f in range( 8 ):
                a = fig.add_subplot( 8, 2, 2 * f + 1 )
                imshow( data[ f, :, : ], cmap = cm.gray )
                a = fig.add_subplot( 8, 2, 2 * f + 2 )
                imshow( vis_act[ f, :, : ], cmap = cm.gray )
            show()
        

print( "training crbm" )

batch = 100
set_size = 60000
num_maps = 40
size_maps = 10
size_image = 28
conv_size = size_image - size_maps + 1

#should these be initialized as random(-1,1) or random[0,1)?
feature_maps = 0.01 * randn( num_maps, conv_size, conv_size )
vis_bias_conv = randn( size_image, size_image )
feature_map_bias = zeros( ( num_maps, size_maps, size_maps ) )

w_grad_conv = zeros( ( num_maps, conv_size, conv_size ) )

#how many biases and who gets one?
hb_grad_conv = zeros( ( num_maps, size_maps, size_maps ) )
vb_grad_conv = zeros( ( size_image, size_image ) )

for epoch in range( max_epoch_2 ):
    for tr_batch in range( set_size // batch ):
        print( "CRBM training epoch ", epoch, "batch number ", tr_batch)
        #should I still make the input units stochastic?
        new_im = images_conv[ batch * tr_batch : batch * ( 1 + tr_batch ), :, : ] / 255.0
        data = new_im > rand( batch, size_image, size_image )
        #initial convolution
        conv_data = array( [ [ signal.convolve2d( data[ j, :, : ], ff( feature_maps[ i, :, : ] ), 'valid' )
                                                                            for i in range( num_maps ) ]
                                                                            for j in range( batch ) ] )
        #print( conv_data.shape, data[ 0, :, : ].shape, feature_maps[ 0, :, : ].shape )
        #get hidden probs and activations                                                                     
        hidden_prob, hidden_act = init_activation_conv( 2 * conv_data + feature_map_bias )

        #reconstruction convolution, take the sum over all feature maps
        #should these be probabilities or activations?
        recon_data = array( [ [ signal.convolve2d( hidden_act[ j, i, :, : ], feature_maps[ i, :, : ], 'full' )
                                                                            for i in range( num_maps ) ]
                                                                            for j in range( batch ) ] ).sum( axis = 1 )
        
        #print( recon_data.shape )
        #reconstruct the visible layer
        recon_prob, recon_act = init_activation_conv_recon( 2 * recon_data )
        
        #reconstruct hidden layer one more time
        #should these be probabilities or activations?
        conv_data_recon = array( [ [ signal.convolve2d( recon_act[ j, :, : ], ff( feature_maps[ i, :, : ] ), 'valid' )
                                                                            for i in range( num_maps ) ]
                                                                            for j in range( batch ) ] )
        recon_hid_prob, recon_hid_act = init_activation_conv( 2 * conv_data_recon + feature_map_bias )
        #update momentum
        if epoch > 4:
            momentum = final_mom
        else:
            momentum = init_mom
        
        #compute gradients and add them to the parameters
        for i in range( num_maps ):
            #print( data.shape, hidden_prob.shape, recon_act.shape, recon_hid_prob.shape )
            #Should these be sum or mean? also should they be the activations or probabilities for each?
            pos_stats = array( [ signal.convolve2d( data[ j, :, : ], ff( hidden_prob[ j, i, :, : ] ), 'valid' )
                                                                                        for j in range( batch ) ] ).sum( axis = 0 )
            neg_stats = array( [ signal.convolve2d( recon_act[ j, :, : ], ff( recon_hid_prob[ j, i, :, : ] ), 'valid' )
                                                                                        for j in range( batch ) ] ).sum( axis = 0 )
            #print( pos_stats.shape, neg_stats.shape, w_grad_conv.shape )
            
            #corrent updates?
            if not i and not tr_batch:
                print( ( ( pos_stats - neg_stats ) / batch ).sum(),
                         ( l2 * feature_maps[ i, :, : ] ).sum(),
                         ( pbias_lambda * w_grad_conv[ i, :, : ] ).sum() )                                                                            
            w_grad_conv[ i, :, : ] = ( momentum * w_grad_conv[ i, :, : ]
                                     + rate * ( ( pos_stats - neg_stats ) / batch
                                     - l2 * feature_maps[ i, :, : ] 
                                     - pbias_lambda * w_grad_conv[ i, :, : ] ) )
        hb_grad_conv = rate * ( ( hidden_prob - recon_hid_prob ).sum( axis = 0 ) / batch - l2 * feature_map_bias )
        #print( w_grad_conv, vb_grad_conv, hb_grad_conv )
        
        #update parameters
        feature_maps += w_grad_conv
        feature_map_bias += hb_grad_conv
        
        print( "train error: ", abs( data - recon_act ).sum() )
        #display example/reconstructions
        if(epoch == 2):
            fig = plt.figure()
            for f in range( 8 ):
                a = fig.add_subplot( 8, 2, 2 * f + 1 )
                imshow( data[ f, :, : ], cmap = cm.gray )
                a = fig.add_subplot( 8, 2, 2 * f + 2 )
                imshow( recon_act[ f, :, : ], cmap = cm.gray )
            plt.show()
        #display feature maps
        if(epoch == 2):
            fig2 = plt.figure()
            for f in range( 40 ):
                a = fig2.add_subplot( 5, 8, f + 1 )
                imshow( feature_maps[ f, :, : ], cmap = cm.gray )
            plt.show()
        #print( hidden_probs.mean() )         
 