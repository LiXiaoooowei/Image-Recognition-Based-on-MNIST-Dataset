"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *

class Layer(object):
    """
       
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        #############################################################
        outputs = inputs.dot(self.weights) + self.bias
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        self.b_grad = np.sum(in_grads, axis = 0)
        self.w_grad = inputs.T.dot(in_grads)
        out_grads = in_grads.dot(self.weights.T)
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        ############################################################# 
        
        ###im2col_index##
        n, c_in, x_h, x_w = inputs.shape
        out_height = (x_h + 2 * self.pad - self.kernel_h) // self.stride + 1
        out_width = (x_w + 2 * self.pad - self.kernel_w) // self.stride + 1
        
        r0 = np.repeat(np.arange(self.kernel_h), self.kernel_w)
        r0 = np.tile(r0, c_in)
        r_bias = self.stride * np.repeat(np.arange(out_height), out_width)
        # shape of r is (c_in * k_h * k_w, o_h * o_w)
        r = r0.reshape(-1, 1) + r_bias.reshape(1, -1)
        r = r.astype(int)

        c0 = np.tile(np.arange(self.kernel_w), self.kernel_h * c_in)
        c_bias = self.stride * np.tile(np.arange(out_width), int(out_height))
        c = c0.reshape(-1, 1) + c_bias.reshape(1, -1) # shape of c is (k_w * k_h * c_in, o_w, o_h)
        c = c.astype(int)

        # shape of d is (c_in * k_w * k_h, 1)
        d = np.repeat(np.arange(c_in), self.kernel_w * self.kernel_h).reshape(-1, 1)
        d = d.astype(int)
        
        ###
        ###im2col###
        inputs_pad = np.pad(inputs, ((0,0), (0,0), (self.pad,self.pad), (self.pad,self.pad)), 'constant')
        cols = inputs_pad[:, d, r, c].transpose(1, 2, 0).reshape(self.kernel_h * self.kernel_w * c_in, -1)
        ###
        
        x_col = cols # shape(n, c_in * k_w * k_h, o_h * o_w)
        w_col = self.weights.reshape((self.out_channel, -1)) # shape(c_out, c_in * k_h * k_w)
        bias = self.bias.reshape(-1, 1)
        outputs = w_col.dot(x_col) + bias # output has shape(n, c_out, o_h * o_w)
        outputs = outputs.reshape((self.out_channel, int(out_height), int(out_width), n))
        outputs= outputs.transpose(3, 0, 1, 2)
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        n, c_out, out_h, out_w = in_grads.shape
        self.b_grad = np.sum(in_grads, axis = (0, 2, 3))
        
        n, c_in, x_h, x_w = inputs.shape
      
        ### im2col_index ##
        out_height = (x_h + 2 * self.pad - self.kernel_h) // self.stride + 1
        out_width = (x_w + 2 * self.pad - self.kernel_w) // self.stride + 1
        
        r0 = np.repeat(np.arange(self.kernel_h), self.kernel_w)
        r0 = np.tile(r0, c_in)

        r_bias = self.stride * np.repeat(np.arange(out_height), out_width)
        # shape of r is (c_in * k_h * k_w, o_h * o_w)
        r = r0.reshape(-1, 1) + r_bias.reshape(1, -1)
        r = r.astype(int)
        
        c0 = np.tile(np.arange(self.kernel_w), self.kernel_h * c_in)
        c_bias = self.stride * np.tile(np.arange(out_width), int(out_height))
        c = c0.reshape(-1, 1) + c_bias.reshape(1, -1) # shape of c is (k_w * k_h * c_in, o_w, o_h)
        c = c.astype(int)

        # shape of d is (c_in * k_w * k_h, 1)
        d = np.repeat(np.arange(c_in), self.kernel_w * self.kernel_h).reshape(-1, 1)
        d = d.astype(int)        
        ###
        
        ### im2col ###
        inputs_pad = np.pad(inputs, ((0,0), (0,0), (self.pad,self.pad), (self.pad,self.pad)), 'constant')
        # shape of x_cols, which is x_hat, is (c_in * k_w * k_h, o_h * o_w, n)
        x_cols = inputs_pad[:, d, r, c].transpose(1, 2, 0).reshape(self.kernel_h * self.kernel_w * c_in, -1) 
        ###
        
        in_grads_reshaped = in_grads.transpose(1, 2, 3, 0).reshape(c_out, -1) # shape(c_out, n * out_h * out_w)
        # in_grads_reshaped has shape (c_out, n * o_h * o_w * c_in * k_h * k_w)
        self.w_grad = in_grads_reshaped.dot(x_cols.T).reshape(self.weights.shape)     
        grads_x_cols = self.weights.reshape(c_out, -1).T.dot(in_grads_reshaped)
        
        ### col2im_indices ###
        x_h_padded, x_w_padded = x_h + 2 * self.pad, x_w + 2 * self.pad
        inputs_padded = np.zeros((n, c_in, x_h_padded, x_w_padded), dtype = np.float64)  
        #print(grads_x_cols.shape)
        grads_x_cols_reshaped = grads_x_cols.reshape(self.kernel_h * self.kernel_w * c_in, -1, n)
        grads_x_cols_reshaped = grads_x_cols_reshaped.transpose(2, 0, 1)
        np.add.at(inputs_padded, (slice(None), d, r, c), grads_x_cols_reshaped)
        if self.pad != 0 :
            out_grads = inputs_padded[:, :, self.pad : -self.pad, self.pad : -self.pad]
        else:
            out_grads = inputs_padded
        ###
        
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        n, c_in, x_h, x_w = inputs.shape
        out_height = (x_h + 2 * self.pad - self.pool_height) // self.stride + 1
        out_width = (x_w + 2 * self.pad - self.pool_width) // self.stride + 1

        if self.pool_type == "max":
            #print(x_set)
            #x_max_indices_x = [np.argmax(sub_x, axis = 0) for sub_x in x_set]
            #out_set = [(sub_x[index_x, np.arange(sub_x.shape[1])]) for (sub_x, index_x) in zip(x_set, x_max_indices_x)]
            #outputs = np.vstack(out_set)
            #outputs = outputs.reshape((c_in, int(out_height), int(out_width), n))
            #outputs= outputs.transpose(3, 0, 1, 2)
            outputs, _ = self.maxpool_forward(inputs)
            
        elif self.pool_type == "avg":
            N, C, H, W = inputs.shape
            w = np.ones((1, 1, self.pool_height, self.pool_width)) / (self.pool_height * self.pool_width)
            num_filters, _, filter_height, filter_width = w.shape
            
            x_split = inputs.reshape(N * C, 1, H, W)

            x_cols = self.im2col(x_split)
            res = w.reshape((1, -1)).dot(x_cols) 

            out = res.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
            outputs = out
        #############################################################
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################             
        if self.pool_type == "max":
            _, cache = self.maxpool_forward(inputs)
            x_cols, x_cols_argmax = cache
            N, C, H, W = inputs.shape
 
            dout_reshaped = in_grads.transpose(2, 3, 0, 1).flatten()
            dx_cols = np.zeros_like(x_cols)
            dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
            dx = self.col2im(dx_cols, (N * C, 1, H, W))
            dx = dx.reshape(inputs.shape)
            out_grads = dx
            
        elif self.pool_type == "avg":
            N, C, H, W = inputs.shape
            x, w = inputs, np.ones((1, 1, self.pool_height, self.pool_width)) / (self.pool_height * self.pool_width)
           
            dout_reshaped = in_grads.transpose(2, 3, 0, 1).flatten()
            x_split = inputs.reshape(N * C, 1, H, W)
            x_cols = self.im2col(x_split)
            dx_cols = np.zeros_like(x_cols)
            dx_cols[:, np.arange(dx_cols.shape[1])] = dout_reshaped / (self.pool_height * self.pool_width)
            dx = self.col2im(dx_cols, (N * C, 1, H, W))
            dx = dx.reshape(inputs.shape)
            out_grads = dx
        #############################################################    
        return out_grads
    
    def im2col_indices(self, X_shape):
        N, C, H, W = X_shape

        out_height = (H + 2 * self.pad - self.pool_height) // self.stride + 1
        out_width = (W + 2 * self.pad - self.pool_width) // self.stride + 1

        i0 = np.repeat(np.arange(self.pool_height), self.pool_width)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(self.pool_width), self.pool_height * C)
        j1 = self.stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), self.pool_height * self.pool_width).reshape(-1, 1)

        return (k, i, j)


    def im2col(self, X):
        """ An implementation of im2col based on some fancy indexing """

        x_padded = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

        k, i, j = self.im2col_indices(X.shape)

        cols = x_padded[:, k, i, j]
        C = X.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(self.pool_height * self.pool_width * C, -1)
        return cols


    def col2im(self, cols, x_shape):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * self.pad, W + 2 * self.pad
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.im2col_indices(x_shape)
        cols_reshaped = cols.reshape(C * self.pool_height * self.pool_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if self.pad == 0:
            return x_padded
        return x_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]

    def maxpool_forward(self, X):
        """
        An implementation of the forward pass for max pooling based on im2col.
        This isn't much faster than the naive version, so it should be avoided if
        possible.
        """
        N, C, H, W = X.shape
 
        out_height = (H - self.pool_height) // self.stride + 1
        out_width = (W - self.pool_width) // self.stride + 1

        x_split = X.reshape(N * C, 1, H, W)
        x_cols = self.im2col(x_split)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

        cache = (x_cols, x_cols_argmax)
        return out, cache


class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = None
        #############################################################
        if self.training:
            np.random.seed(self.seed)
            self.mask = np.random.binomial(1, 1 - self.ratio, size = inputs.shape)
            outputs = inputs * self.mask * (1 / (1 - self.ratio))
        else:
            outputs = inputs
            
        outputs = outputs.astype(inputs.dtype, copy = False)
        #############################################################       
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        #############################################################
        if self.training:
            out_grads = in_grads * self.mask * (1 / (1 - self.ratio))
        else:
            out_grads = in_grads 
        #############################################################
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
