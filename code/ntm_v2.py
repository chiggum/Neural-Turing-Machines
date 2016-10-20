import theano
import theano.tensor as T
from theano import shared, function, config
from rmsprop_orig import *
from print_info import *

import numpy as np
from collections import OrderedDict
from datetime import datetime

np.set_printoptions(threshold='nan')

def CAS(b_m_t_1, k_t, beta_t, eps=1e-6):
    mem_norm = b_m_t_1.norm(2, axis=1)+eps
    k_t_norm = k_t.norm(2)
    w_t_c = T.nnet.softmax((((b_m_t_1.dot(k_t))/mem_norm)/k_t_norm)*beta_t)
    return w_t_c

def Interpolate(w_t_c, b_w_t_1, g_t):
    w_t_g = g_t * w_t_c + (1-g_t) * b_w_t_1
    return w_t_g

def ShiftConv(w_t_g, s_t, N, num_shifts):
    # pad = (num_shifts//2, (num_shifts-1)//2)
    # w_t_g_pd_ = T.concatenate([w_t_g[(-pad[0]-1):-1], w_t_g, w_t_g[:(pad[1])]])
    # w_t_g_pd = w_t_g_pd_.dimshuffle('x','x','x', 0)
    # filter = s_t.dimshuffle('x', 'x', 'x', 0)
    # convolution = T.nnet.conv2d(w_t_g_pd, filter,
    # input_shape=(1, 1, 1, N + pad[0] + pad[1]),
    # filter_shape=(1, 1, 1, num_shifts),
    # subsample=(1, 1),
    # border_mode='valid')
    # w_t_s = convolution[0, 0, 0, :]
    shift = 2.*s_t-1.
    Z = T.mod(shift+N, N)
    simj = 1 - (Z - T.floor(Z))
    imj = T.mod(T.arange(N) + T.iround(T.floor(Z)),N)
    w_t_g_roll_1 = T.roll(w_t_g, -T.iround(T.floor(Z)))
    w_t_g_roll_2 = T.roll(w_t_g, -(T.iround(T.floor(Z))+1))
    w_t_s = w_t_g_roll_1*simj + w_t_g_roll_2*(1-simj)
    return w_t_s

def Sharpen(w_t_s, gamma_t):
    w_t_f_tilda = w_t_s ** gamma_t
    w_t_f = w_t_f_tilda/T.sum(w_t_f_tilda)
    return w_t_f
    
def ReadVec(b_m, b_w_mat):
    return b_w_mat.dot(b_m)

# Caller must make sure that dimensions
# are properly set
def MemoryErase(b_m, b_w, e_t):
    return -1.*b_m*(b_w.dot(e_t))

# Caller must make sure that dimensions
# are properly set
def MemoryAdd(b_w, a_t):
    return b_w.dot(a_t)

class NTM:
    def __init__(self, ext_in_dim, ext_out_dim, mem_shape, n_read_heads, n_write_heads, hidden_dim, num_shifts=3):
        self.ext_in_dim = ext_in_dim
        self.ext_out_dim = ext_out_dim
        self.mem_shape = mem_shape
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.hidden_dim = hidden_dim
        self.num_shifts = num_shifts
        
        params = OrderedDict()
        sqrt23 = np.sqrt(6)
        rng = sqrt23/np.sqrt(hidden_dim)
        params['b_hidden'] = np.random.uniform(-rng,rng, hidden_dim)
        rng = sqrt23/np.sqrt(np.prod(mem_shape))
        params['b_m'] = np.random.uniform(-rng,rng, mem_shape)
        #params['b_m'] = np.zeros(mem_shape) + 1e-6
        rng = sqrt23/np.sqrt(n_read_heads*mem_shape[0])
        params['b_r_w_mat'] = np.random.uniform(-rng,rng, (n_read_heads, mem_shape[0]))
        #params['b_r_w_mat'] = np.eye(n_read_heads, mem_shape[0])*1.0
        rng = sqrt23/np.sqrt(n_write_heads*mem_shape[0])
        params['b_w_w_mat'] = np.random.uniform(-rng,rng, (n_write_heads, mem_shape[0])) 
        #params['b_w_w_mat'] = np.eye(n_write_heads, mem_shape[0])*1.0 
        rng = sqrt23/np.sqrt(hidden_dim+ext_in_dim)
        params['W_e_i_h'] = np.random.uniform(-rng,rng, (hidden_dim, ext_in_dim))
        params['b_e_i_h'] = (np.random.uniform(-rng,rng, hidden_dim) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_read_heads*mem_shape[1])
        params['W_r_h'] = np.random.uniform(-rng,rng, (hidden_dim, n_read_heads*mem_shape[1]))
        params['b_r_h'] = (np.random.uniform(-rng,rng, hidden_dim) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+ext_out_dim)
        params['W_h_e_o'] = np.random.uniform(-rng,rng, (ext_out_dim, hidden_dim))
        params['b_h_e_o'] = (np.random.uniform(-rng,rng, ext_out_dim) ) * 0.
        
        rng = sqrt23/np.sqrt(hidden_dim+n_read_heads*mem_shape[1])
        params['r_W_h_k_o'] = np.random.uniform(-rng,rng, (n_read_heads*mem_shape[1], hidden_dim)) 
        params['r_b_h_k_o'] = (np.random.uniform(-rng,rng, n_read_heads*mem_shape[1]) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_read_heads*1)
        params['r_W_h_beta_o'] = np.random.uniform(-rng,rng, (n_read_heads*1, hidden_dim)) 
        params['r_b_h_beta_o'] = (np.random.uniform(-rng,rng, n_read_heads*1) ) * 0.
        params['r_W_h_g_o'] = np.random.uniform(-rng,rng, (n_read_heads*1, hidden_dim)) 
        params['r_b_h_g_o'] = (np.random.uniform(-rng,rng, n_read_heads*1) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_read_heads*num_shifts)
        params['r_W_h_s_o'] = np.random.uniform(-rng,rng, (n_read_heads*num_shifts, hidden_dim)) 
        params['r_b_h_s_o'] = (np.random.uniform(-rng,rng, n_read_heads*num_shifts) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_read_heads*1)
        params['r_W_h_gamma_o'] = np.random.uniform(-rng,rng, (n_read_heads*1, hidden_dim)) 
        params['r_b_h_gamma_o'] = (np.random.uniform(-rng,rng, n_read_heads*1) ) * 0.
        
        rng = sqrt23/np.sqrt(hidden_dim+n_write_heads*mem_shape[1])
        params['w_W_h_k_o'] = np.random.uniform(-rng,rng, (n_write_heads*mem_shape[1], hidden_dim)) 
        params['w_b_h_k_o'] = (np.random.uniform(-rng,rng, n_write_heads*mem_shape[1]) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_write_heads*1)
        params['w_W_h_beta_o'] = np.random.uniform(-rng,rng, (n_write_heads*1, hidden_dim)) 
        params['w_b_h_beta_o'] = (np.random.uniform(-rng,rng, n_write_heads*1) ) * 0.
        params['w_W_h_g_o'] = np.random.uniform(-rng,rng, (n_write_heads*1, hidden_dim)) 
        params['w_b_h_g_o'] = (np.random.uniform(-rng,rng, n_write_heads*1) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_write_heads*num_shifts)
        params['w_W_h_s_o'] = np.random.uniform(-rng,rng, (n_write_heads*num_shifts, hidden_dim)) 
        params['w_b_h_s_o'] = (np.random.uniform(-rng,rng, n_write_heads*num_shifts) ) * 0.
        rng = sqrt23/np.sqrt(hidden_dim+n_write_heads*1)
        params['w_W_h_gamma_o'] = np.random.uniform(-rng,rng, (n_write_heads*1, hidden_dim)) 
        params['w_b_h_gamma_o'] = (np.random.uniform(-rng,rng, n_write_heads*1) ) * 0.
        
        rng = sqrt23/np.sqrt(hidden_dim+n_write_heads*mem_shape[1])
        params['W_h_er_o'] = np.random.uniform(-rng,rng, (n_write_heads*mem_shape[1], hidden_dim)) 
        params['b_h_er_o'] = (np.random.uniform(-rng,rng, n_write_heads*mem_shape[1]) ) * 0.
        params['W_h_a_o'] = np.random.uniform(-rng,rng, (n_write_heads*mem_shape[1], hidden_dim)) 
        params['b_h_a_o'] = (np.random.uniform(-rng,rng, n_write_heads*mem_shape[1]) ) * 0.
        
        tparams = OrderedDict()
        tparams['b_hidden'] = shared(name='b_hidden', value=params['b_hidden'].astype(config.floatX))
        tparams['b_m'] = shared(name='b_m', value=params['b_m'].astype(config.floatX))
        tparams['b_r_w_mat'] = shared(name='b_r_w_mat', value=params['b_r_w_mat'].astype(config.floatX))
        tparams['b_w_w_mat'] = shared(name='b_w_w_mat', value=params['b_w_w_mat'].astype(config.floatX))
        tparams['W_e_i_h'] = shared(name='W_e_i_h', value=params['W_e_i_h'].astype(config.floatX))
        tparams['b_e_i_h'] = shared(name='b_e_i_h', value=params['b_e_i_h'].astype(config.floatX))
        tparams['W_r_h'] = shared(name='W_r_h', value=params['W_r_h'].astype(config.floatX))
        tparams['b_r_h'] = shared(name='b_r_h', value=params['b_r_h'].astype(config.floatX))
        tparams['W_h_e_o'] = shared(name='W_h_e_o', value=params['W_h_e_o'].astype(config.floatX))
        tparams['b_h_e_o'] = shared(name='b_h_e_o', value=params['b_h_e_o'].astype(config.floatX))
        
        tparams['r_W_h_k_o'] = shared(name='r_W_h_k_o', value=params['r_W_h_k_o'].astype(config.floatX))
        tparams['r_b_h_k_o'] = shared(name='r_b_h_k_o', value=params['r_b_h_k_o'].astype(config.floatX))
        tparams['r_W_h_beta_o'] = shared(name='r_W_h_beta_o', value=params['r_W_h_beta_o'].astype(config.floatX))
        tparams['r_b_h_beta_o'] = shared(name='r_b_h_beta_o', value=params['r_b_h_beta_o'].astype(config.floatX))
        tparams['r_W_h_g_o'] = shared(name='r_W_h_g_o', value=params['r_W_h_g_o'].astype(config.floatX))
        tparams['r_b_h_g_o'] = shared(name='r_b_h_g_o', value=params['r_b_h_g_o'].astype(config.floatX))
        tparams['r_W_h_s_o'] = shared(name='r_W_h_s_o', value=params['r_W_h_s_o'].astype(config.floatX))
        tparams['r_b_h_s_o'] = shared(name='r_b_h_s_o', value=params['r_b_h_s_o'].astype(config.floatX))
        tparams['r_W_h_gamma_o'] = shared(name='r_W_h_gamma_o', value=params['r_W_h_gamma_o'].astype(config.floatX))
        tparams['r_b_h_gamma_o'] = shared(name='r_b_h_gamma_o', value=params['r_b_h_gamma_o'].astype(config.floatX))
        
        tparams['w_W_h_k_o'] = shared(name='w_W_h_k_o', value=params['w_W_h_k_o'].astype(config.floatX))
        tparams['w_b_h_k_o'] = shared(name='w_b_h_k_o', value=params['w_b_h_k_o'].astype(config.floatX))
        tparams['w_W_h_beta_o'] = shared(name='w_W_h_beta_o', value=params['w_W_h_beta_o'].astype(config.floatX))
        tparams['w_b_h_beta_o'] = shared(name='w_b_h_beta_o', value=params['w_b_h_beta_o'].astype(config.floatX))
        tparams['w_W_h_g_o'] = shared(name='w_W_h_g_o', value=params['w_W_h_g_o'].astype(config.floatX))
        tparams['w_b_h_g_o'] = shared(name='w_b_h_g_o', value=params['w_b_h_g_o'].astype(config.floatX))
        tparams['w_W_h_s_o'] = shared(name='w_W_h_s_o', value=params['w_W_h_s_o'].astype(config.floatX))
        tparams['w_b_h_s_o'] = shared(name='w_b_h_s_o', value=params['w_b_h_s_o'].astype(config.floatX))
        tparams['w_W_h_gamma_o'] = shared(name='w_W_h_gamma_o', value=params['w_W_h_gamma_o'].astype(config.floatX))
        tparams['w_b_h_gamma_o'] = shared(name='w_b_h_gamma_o', value=params['w_b_h_gamma_o'].astype(config.floatX))
        
        tparams['W_h_er_o'] = shared(name='W_h_er_o', value=params['W_h_er_o'].astype(config.floatX))
        tparams['b_h_er_o'] = shared(name='b_h_er_o', value=params['b_h_er_o'].astype(config.floatX))
        tparams['W_h_a_o'] = shared(name='W_h_a_o', value=params['W_h_a_o'].astype(config.floatX))
        tparams['b_h_a_o'] = shared(name='b_h_a_o', value=params['b_h_a_o'].astype(config.floatX))
        
        self.tparams = tparams
        
        self.all_tparams = []
        self.all_tparams_ind = OrderedDict()
        for i, (key, value) in enumerate(self.tparams.iteritems()):
            self.all_tparams.append(value)
            self.all_tparams_ind[key]=i
        
        self.opt = rmsprop(self.all_tparams)
        
        self.__theano_build__()
    
    def __theano_build__(self):
        b_hidden = self.tparams['b_hidden']
        b_m = self.tparams['b_m']
        b_r_w_mat = self.tparams['b_r_w_mat']
        b_w_w_mat = self.tparams['b_w_w_mat']
        W_e_i_h = self.tparams['W_e_i_h']
        b_e_i_h = self.tparams['b_e_i_h']
        W_r_h = self.tparams['W_r_h']
        b_r_h = self.tparams['b_r_h']
        W_h_e_o = self.tparams['W_h_e_o']
        b_h_e_o = self.tparams['b_h_e_o']
        
        r_W_h_k_o = self.tparams['r_W_h_k_o']
        r_b_h_k_o = self.tparams['r_b_h_k_o']
        r_W_h_beta_o = self.tparams['r_W_h_beta_o']
        r_b_h_beta_o = self.tparams['r_b_h_beta_o']
        r_W_h_g_o = self.tparams['r_W_h_g_o']
        r_b_h_g_o = self.tparams['r_b_h_g_o']
        r_W_h_s_o = self.tparams['r_W_h_s_o']
        r_b_h_s_o = self.tparams['r_b_h_s_o']
        r_W_h_gamma_o = self.tparams['r_W_h_gamma_o']
        r_b_h_gamma_o = self.tparams['r_b_h_gamma_o']
        
        w_W_h_k_o = self.tparams['w_W_h_k_o']
        w_b_h_k_o = self.tparams['w_b_h_k_o']
        w_W_h_beta_o = self.tparams['w_W_h_beta_o']
        w_b_h_beta_o = self.tparams['w_b_h_beta_o']
        w_W_h_g_o = self.tparams['w_W_h_g_o']
        w_b_h_g_o = self.tparams['w_b_h_g_o']
        w_W_h_s_o = self.tparams['w_W_h_s_o']
        w_b_h_s_o = self.tparams['w_b_h_s_o']
        w_W_h_gamma_o = self.tparams['w_W_h_gamma_o']
        w_b_h_gamma_o = self.tparams['w_b_h_gamma_o']
        
        W_h_er_o = self.tparams['W_h_er_o']
        b_h_er_o = self.tparams['b_h_er_o']
        W_h_a_o = self.tparams['W_h_a_o']
        b_h_a_o = self.tparams['b_h_a_o']
        
        mem_shape = self.mem_shape
        ext_out_dim = self.ext_out_dim
        n_read_heads = self.n_read_heads
        n_write_heads = self.n_write_heads
        num_shifts = self.num_shifts
        
        X = T.matrix('X')
        Y = T.matrix('Y')
        
        def _b_w_mat_update(b_w_t, k_t, beta_t, g_t, s_t, gamma_t, b_m):           
            w_t_c = (CAS(b_m, k_t, beta_t)).flatten()
            w_t_g = (Interpolate(w_t_c, b_w_t, g_t)).flatten()
            w_t_s = (ShiftConv(w_t_g, s_t, mem_shape[0], num_shifts)).flatten()
            w_t_f = (Sharpen(w_t_s, gamma_t)).flatten()
            return w_t_f
        
        def _b_m_update(b_w_t, e_t, a_t, b_m_t_1, b_m_orig):
            e_t_tilda = e_t.reshape((1, mem_shape[1]))
            a_t_tilda = a_t.reshape((1, mem_shape[1]))
            b_w_t_tilda = b_w_t.reshape((mem_shape[0], 1))
            b_m_t_erase = MemoryErase(b_m_orig, b_w_t_tilda, e_t_tilda)
            b_m_t_add = MemoryAdd(b_w_t_tilda, a_t_tilda)
            b_m_t = b_m_t_1 + b_m_t_erase + b_m_t_add
            return b_m_t
        
        def _step(X_t, b_m_t_1, b_r_w_mat_t_1, b_w_w_mat_t_1, b_hidden_t_1,
                    W_e_i_h_, b_e_i_h_, W_r_h_, b_r_h_, W_h_e_o_, b_h_e_o_, 
                    r_W_h_k_o_, r_b_h_k_o_, r_W_h_beta_o_, r_b_h_beta_o_, r_W_h_g_o_, r_b_h_g_o_,
                    r_W_h_s_o_, r_b_h_s_o_, r_W_h_gamma_o_, r_b_h_gamma_o_,
                    w_W_h_k_o_, w_b_h_k_o_, w_W_h_beta_o_, w_b_h_beta_o_, w_W_h_g_o_, w_b_h_g_o_,
                    w_W_h_s_o_, w_b_h_s_o_, w_W_h_gamma_o_, w_b_h_gamma_o_, W_h_er_o_, b_h_er_o_,
                    W_h_a_o_, b_h_a_o_):
            
            er_o_t = (T.nnet.sigmoid(W_h_er_o_.dot(b_hidden_t_1)+b_h_er_o_)).reshape((n_write_heads, mem_shape[1]))
            #a_o_t = (T.nnet.sigmoid(W_h_a_o_.dot(h_t)+b_h_a_o_)).reshape((n_write_heads, mem_shape[1]))
            #a_o_t = (W_h_a_o_.dot(h_t)+b_h_a_o_).reshape((n_write_heads, mem_shape[1]))
            a_o_t = (T.clip(W_h_a_o_.dot(b_hidden_t_1)+b_h_a_o_, -1., 1.)).reshape((n_write_heads, mem_shape[1]))
            
            b_m_t_seq, updates_m_t = theano.scan(
                _b_m_update,
                non_sequences = b_m_t_1,
                sequences = [b_w_w_mat_t_1, er_o_t, a_o_t],
                outputs_info=dict(initial=b_m_t_1)
            )
            b_m_t = b_m_t_seq[-1]
            
            r_t_mat = ReadVec(b_m_t, b_r_w_mat_t_1)
            r_t = r_t_mat.flatten()
            #h_t = T.nnet.sigmoid(W_e_i_h_.dot(X_t)+b_e_i_h_ + W_r_h_.dot(r_t)+b_r_h_)
            h_t = T.nnet.relu(W_e_i_h_.dot(X_t)+b_e_i_h_ + W_r_h_.dot(r_t)+b_r_h_)
            b_hidden_t = h_t
            e_o_t = T.nnet.sigmoid(W_h_e_o_.dot(h_t)+b_h_e_o_)
            
            #r_k_o_t = (T.nnet.relu(r_W_h_k_o_.dot(h_t)+r_b_h_k_o_)).reshape((n_read_heads, mem_shape[1]))
            r_k_o_t = (T.clip(r_W_h_k_o_.dot(h_t)+r_b_h_k_o_, -1., 1.)).reshape((n_read_heads, mem_shape[1]))
            #r_beta_o_t = (T.exp(r_W_h_beta_o_.dot(h_t)+r_b_h_beta_o_)).flatten()
            r_beta_o_t = (T.nnet.relu(r_W_h_beta_o_.dot(h_t)+r_b_h_beta_o_)).flatten()
            r_g_o_t = (T.nnet.sigmoid(r_W_h_g_o_.dot(h_t)+r_b_h_g_o_)).flatten()
            r_s_o_t = T.nnet.sigmoid((r_W_h_s_o_.dot(h_t)+r_b_h_s_o_)).flatten()
            #r_s_o_t = T.nnet.softmax((r_W_h_s_o_.dot(h_t)+r_b_h_s_o_).reshape((n_read_heads, num_shifts)))
            #r_gamma_o_t = (T.log(T.exp(r_W_h_gamma_o_.dot(h_t)+r_b_h_gamma_o_)+1.)+1.).flatten()
            r_gamma_o_t = (T.nnet.relu(r_W_h_gamma_o_.dot(h_t)+r_b_h_gamma_o_)+1.).flatten()
            
            w_k_o_t = (T.clip(w_W_h_k_o_.dot(h_t)+w_b_h_k_o_, -1., 1.)).reshape((n_write_heads, mem_shape[1]))
            #w_k_o_t = (T.nnet.relu(w_W_h_k_o_.dot(h_t)+w_b_h_k_o_)).reshape((n_write_heads, mem_shape[1]))
            #w_beta_o_t = (T.exp(w_W_h_beta_o_.dot(h_t)+w_b_h_beta_o_)).flatten()
            w_beta_o_t = (T.nnet.relu(w_W_h_beta_o_.dot(h_t)+w_b_h_beta_o_)).flatten()
            w_g_o_t = (T.nnet.sigmoid(w_W_h_g_o_.dot(h_t)+w_b_h_g_o_)).flatten()
            w_s_o_t = T.nnet.sigmoid((w_W_h_s_o_.dot(h_t)+w_b_h_s_o_)).flatten()
            #w_s_o_t = T.nnet.softmax((w_W_h_s_o_.dot(h_t)+w_b_h_s_o_).reshape((n_write_heads, num_shifts)))
            #w_gamma_o_t = (T.log(T.exp(w_W_h_gamma_o_.dot(h_t)+w_b_h_gamma_o_)+1.)+1.).flatten()
            w_gamma_o_t = (T.nnet.relu(w_W_h_gamma_o_.dot(h_t)+w_b_h_gamma_o_)+1.).flatten()
            
            # er_o_t = (T.nnet.sigmoid(W_h_er_o_.dot(h_t)+b_h_er_o_)).reshape((n_write_heads, mem_shape[1]))
            # #a_o_t = (T.nnet.sigmoid(W_h_a_o_.dot(h_t)+b_h_a_o_)).reshape((n_write_heads, mem_shape[1]))
            # #a_o_t = (W_h_a_o_.dot(h_t)+b_h_a_o_).reshape((n_write_heads, mem_shape[1]))
            # a_o_t = (T.clip(W_h_a_o_.dot(h_t)+b_h_a_o_, -1., 1.)).reshape((n_write_heads, mem_shape[1]))
            
            r_w_t_f_mat, updates_r_w_t = theano.scan(
                _b_w_mat_update,
                non_sequences = [b_m_t],
                sequences = [b_r_w_mat_t_1, r_k_o_t, r_beta_o_t, r_g_o_t, r_s_o_t, r_gamma_o_t],
                outputs_info=None
            )
            
            w_w_t_f_mat, updates_w_w_t = theano.scan(
                _b_w_mat_update,
                non_sequences = [b_m_t],
                sequences = [b_w_w_mat_t_1, w_k_o_t, w_beta_o_t, w_g_o_t, w_s_o_t, w_gamma_o_t],
                outputs_info=None
            )
            
            # b_m_t_seq, updates_m_t = theano.scan(
            #     _b_m_update,
            #     non_sequences = b_m_t_1,
            #     sequences = [w_w_t_f_mat, er_o_t, a_o_t],
            #     outputs_info=dict(initial=b_m_t_1)
            # )
            
            b_r_w_mat_t = r_w_t_f_mat
            b_w_w_mat_t = w_w_t_f_mat
            # b_m_t = b_m_t_seq[-1]
            
            return [e_o_t, b_m_t, b_r_w_mat_t, b_w_w_mat_t, b_hidden_t, r_t_mat, a_o_t]
        
        #b_r_w_mat_soft = b_r_w_mat
        b_r_w_mat_soft = T.nnet.softmax(b_r_w_mat)
        #b_w_w_mat_soft = b_w_w_mat
        b_w_w_mat_soft = T.nnet.softmax(b_w_w_mat)
        
        [external_output, b_m_seq, b_r_w_mat_seq, b_w_w_mat_seq, b_hidden_seq,
         debug_out_1, debug_out_2], all_updates = theano.scan(
            _step,
            sequences=[X],
            non_sequences=[W_e_i_h, b_e_i_h, W_r_h, b_r_h, W_h_e_o, b_h_e_o, 
                    r_W_h_k_o, r_b_h_k_o, r_W_h_beta_o, r_b_h_beta_o, r_W_h_g_o, r_b_h_g_o,
                    r_W_h_s_o, r_b_h_s_o, r_W_h_gamma_o, r_b_h_gamma_o,
                    w_W_h_k_o, w_b_h_k_o, w_W_h_beta_o, w_b_h_beta_o, w_W_h_g_o, w_b_h_g_o,
                    w_W_h_s_o, w_b_h_s_o, w_W_h_gamma_o, w_b_h_gamma_o, W_h_er_o, b_h_er_o,
                    W_h_a_o, b_h_a_o],
            outputs_info=[None, dict(initial=b_m), dict(initial=b_r_w_mat_soft),
                             dict(initial=b_w_w_mat_soft), dict(initial=b_hidden), None, None]
        )
        
        o_err = T.mean(T.nnet.binary_crossentropy(external_output, Y))
        hamm_err = T.sum(T.neq(T.ge(external_output, 0.5)*1.0, Y))
        thresh_prediction = T.ge(external_output, 0.5)
        
        dtparams = [T.clip(g, -10, 10) for g in T.grad(o_err, wrt=self.all_tparams)]
        grad_abs_sum = T.sum(0)
        grad_sum = T.sum(0)
        for g in dtparams:
            grad_abs_sum += T.sum(T.abs_(g))
            grad_sum += T.sum(g)
        
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar("momentum")
        rmsprop_updates = self.opt.graves_rmsprop_updates(self.all_tparams, dtparams, learning_rate, momentum)

        self.prediction = function([X], [external_output, thresh_prediction])
        self.prediction_plus_err = function([X, Y], [external_output, thresh_prediction, 
                                                    o_err, hamm_err, debug_out_1, debug_out_2, b_w_w_mat_seq, b_r_w_mat_seq])
        self.loss = function([X, Y], [o_err, hamm_err])
        self.rmsprop_step = function(
            [X, Y, learning_rate, momentum],
            [o_err, hamm_err, external_output, W_h_e_o, b_h_e_o],
            updates=rmsprop_updates)
    
    def get_model_params_to_save(self):
        model_params = OrderedDict()
        for i, (key,value) in enumerate(self.tparams.iteritems()):
            model_params[key] = value.get_value()
        return model_params
    
    def load_model_params(self, model_params):
        for i, (key,value) in enumerate(self.tparams.iteritems()):
            self.tparams[key].set_value(model_params[key])

def save_model_parameters(outfile, model):
    model_params = model.get_model_params_to_save()
    np.savez(outfile, model_params=model_params)
    print "Saved model parameters to %s." % outfile

def save_info(outfile, X, Y, info):
    my_info = OrderedDict()
    my_info['X'] = X
    my_info['Y'] = Y
    my_info['prediction'] = info[0]
    my_info['thresh_prediction'] = info[1]
    my_info['ce_error'] = info[2]
    my_info['hm_error'] = info[3]
    my_info['reads'] = info[4]
    my_info['adds'] = info[5]
    my_info['read_weights'] = info[7]
    my_info['write_weights'] = info[6]
    np.savez(outfile, info=my_info)
    print "Saved info to %s." % outfile

def load_model_parameters(path, model):
    npzfile = np.load(path)
    model_params = npzfile['model_params'].tolist()
    model.load_model_params(model_params)
    print "Loaded model parameters from %s." % (path)

def run_ntm_copy():
    np.random.seed(2)
    task_name = "COPY"
    ext_in_dim = 8+2
    ext_out_dim = 8
    mem_shape = (128, 20)
    n_read_heads = 1
    n_write_heads = 1
    hidden_dim = 100
    model = NTM(ext_in_dim, ext_out_dim, mem_shape, n_read_heads, n_write_heads, hidden_dim)
    max_iter = 100000
    max_len = 5
    min_len = 1
    learning_rate = 1e-3
    momentum = 0.5
    printFreq = 100
    max_iter = 1000000
    save_model_param_after = 5000
    to_test = True
    test_path = "../model/ntm2-COPY-5-2016-03-13-02-32-14.npz"
    load_path = ""
    if to_test:
        load_model_parameters(test_path, model)
    
    if load_path != "":
        load_model_parameters(load_path, model)

    time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    deb_f = None
    if not to_test:
        debug_output_file = ("../plots/ntm2_learning_curve-%s-%d-%s.txt" % (task_name, max_len, time_now))
        deb_f = open(debug_output_file, "w")
    else:
        min_len = 10
        max_len = 120
        max_iter = 20

    for iter in range(max_iter):
        # Generate input
        mylen = np.random.randint(min_len, max_len+1)
        X = np.zeros((2*mylen+2, ext_in_dim)).astype(np.float32)
        Y = np.zeros((2*mylen+2, ext_out_dim)).astype(np.float32)
        X[0, ext_out_dim] = 1
        X[mylen+1, ext_out_dim+1] = 1
        X[1:(mylen+1), :(ext_out_dim)] = (np.random.randint(0, 100000, (mylen, ext_out_dim))%2)*1.0
        Y[(mylen+2):(2*mylen+2), :] = X[1:(mylen+1), :(ext_out_dim)]
        
        if not to_test:
            # take rms prop step
            err = model.rmsprop_step(X, Y, learning_rate, momentum)
            if iter%printFreq==0:
                print "Iter: %d Len: %d - CE Error: %f HM Error: %f" % (iter, mylen, err[0], err[1])
                deb_f.write(("%d,%d,%f,%f\n" % (iter, mylen, err[0], err[1])))
                # if hamming distance is zero
                if err[1] == 0:
                    time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    save_model_parameters("../model/ntm2-%s-%d-%s.npz" % (task_name, max_len, time_now), model)
                elif iter%save_model_param_after == 0:
                    time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    save_model_parameters("../model/ntm2_reg_save_-%s-%d-%s.npz" % (task_name, max_len, time_now), model)
        else:
            info = model.prediction_plus_err(X, Y)
            print "Iter", iter, "Len:", mylen, "Error: ", info[2], info[3]
            time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_info("../plots/ntm2-info-%s-%d-%d-%d-%s.npz" % (task_name, min_len, max_len, mylen, time_now), X, Y, info)
            print_info("../plots/ntm2-info-%s-%d-%d-%d-%s.npz" % (task_name, min_len, max_len, mylen, time_now), 
                                        ext_in_dim, ext_out_dim, mem_shape, n_read_heads, n_write_heads, task_name)

    if not to_test:
        deb_f.close()
        print_learning_curve(debug_output_file, task_name, 20)

run_ntm_copy()

# def run_ntm_psort():
#     np.random.seed(2)
#     task_name = "PRIORITY_SORT"
#     ext_in_dim = 7+1+2
#     ext_out_dim = 7+1
#     mem_shape = (128, 20)
#     n_heads = 8
#     hidden_dim = 512
#     model = NTM(ext_in_dim, ext_out_dim, mem_shape, n_heads, hidden_dim)
#     max_iter = 100000
#     in_len = 20
#     out_len = 16
#     learning_rate = 3*1e-5
#     momentum = 0.9
#     printFreq = 100
#     max_iter = 1000000
#     to_test = False
#     to_debug = True
#     test_path = "../model/ntm2-COPY-20-2016-03-04-17-06-51.npz"

#     if to_test:
#         load_model_parameters(test_path, model)

#     time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     deb_f = None
#     if to_debug:
#         if not to_test:
#             debug_output_file = ("../plots/ntm_learning_curve-%s-%d-%s.txt" % (task_name, out_len, time_now))
#             deb_f = open(debug_output_file, "w")
#         else:
#             in_len = 20
#             out_len = 16
#             max_iter = 20

#     for iter in range(max_iter):
#         # Generate input
#         X = np.zeros((in_len+out_len+2, ext_in_dim))
#         Y = np.zeros((in_len+out_len+2, ext_out_dim))
#         X[0, ext_out_dim] = 1
#         X[in_len+1, ext_out_dim+1] = 1
#         X[1:(in_len+1), :(ext_out_dim-1)] = (np.random.randint(0, 100000, (in_len, ext_out_dim-1))%2)*1.0
#         prioritities = (np.random.uniform(-1., 1., (in_len)))*1.0
#         X[1:(in_len+1), (ext_out_dim-1)] = prioritities
#         sorted_priorities_i = np.argsort(prioritities)[:out_len]
#         Y[(in_len+2):(in_len+out_len+2), :] = X[1+sorted_priorities_i, :(ext_out_dim)]
        
#         if not to_test:
#             # take rms prop step
#             err = model.rmsprop_step(X, Y, learning_rate, momentum)
#             if iter%printFreq==0:
#                 print "Iter: %d Len: %d - CE Error: %f HM Error: %f" % (iter, out_len, err[0], err[1])
#                 if to_debug:
#                     print "Prediction: "
#                     print err[2].flatten()[:10]
#                     print "Gradient Abs Sum: "
#                     print err[3]
#                     print "Gradient Sum: "
#                     print err[4]
#                 else:
#                     deb_f.write(("%d,%d,%f,%f\n" % (iter, out_len, err[0], err[1])))
#                 # if hamming distance is zero
#                 if err[1] == 0:
#                     time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#                     save_model_parameters("../model/ntm-%s-%d-%s.npz" % (task_name, in_len, time_now), model)
#         else:
#             info = model.prediction_plus_err(X, Y)
#             print "Iter", iter, "Len:", out_len, "Error: ", info[2], info[3]
#             time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#             save_info("../plots/ntm-info-%s-%d-%d-%d-%s.npz" % (task_name, out_len, in_len, out_len, time_now), X, Y, info)
#             print_info("../plots/ntm-info-%s-%d-%d-%d-%s.npz" % (task_name, out_len, in_len, out_len, time_now), 
#                                         ext_in_dim, ext_out_dim, mem_shape, n_heads, task_name)

#     if not to_test and not to_debug:
#         deb_f.close()
#         print_learning_curve(debug_output_file, task_name)

# run_ntm_psort()