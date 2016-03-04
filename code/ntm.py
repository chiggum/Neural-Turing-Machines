import theano
import theano.tensor as T
from theano import shared, function, config
from rmsprop_orig import *
from print_info import *

import numpy as np
from collections import OrderedDict
from datetime import datetime

def CAS(b_m_t_1, k_t, beta_t):
    mem_norm = b_m_t_1.norm(2, axis=1)
    k_t_norm = k_t.norm(2)
    w_t_c = T.nnet.softmax((((b_m_t_1.dot(k_t))/mem_norm)/k_t_norm)*beta_t)
    return w_t_c

def Interpolate(w_t_c, b_w_t_1, g_t):
    w_t_g = g_t * w_t_c + (1-g_t) * b_w_t_1
    return w_t_g

def ShiftConv(w_t_g, s_t, N):
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
    def __init__(self, ext_in_dim, ext_out_dim, mem_shape, n_heads, hidden_dim):
        self.ext_in_dim = ext_in_dim
        self.ext_out_dim = ext_out_dim
        self.mem_shape = mem_shape
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        params = OrderedDict()
        params['b_m'] = np.random.uniform(0.,1., mem_shape) - 0.5
        params['b_w_mat'] = np.random.uniform(0.,1., (n_heads, mem_shape[0])) - 0.5
        params['W_e_i_h'] = np.random.uniform(0.,1., (hidden_dim, ext_in_dim)) - 0.5
        params['b_e_i_h'] = (np.random.uniform(0.,1., hidden_dim) - 0.5) * 0.
        params['W_r_h'] = np.random.uniform(0.,1., (hidden_dim, n_heads*mem_shape[1])) - 0.5
        params['b_r_h'] = (np.random.uniform(0.,1., hidden_dim) - 0.5) * 0.
        params['W_h_e_o'] = np.random.uniform(0.,1., (ext_out_dim, hidden_dim)) - 0.5
        params['b_h_e_o'] = (np.random.uniform(0.,1., ext_out_dim) - 0.5) * 0.
        params['W_h_k_o'] = np.random.uniform(0.,1., (n_heads*mem_shape[1], hidden_dim)) - 0.5
        params['b_h_k_o'] = (np.random.uniform(0.,1., n_heads*mem_shape[1]) - 0.5) * 0.
        params['W_h_beta_o'] = np.random.uniform(0.,1., (n_heads*1, hidden_dim)) - 0.5
        params['b_h_beta_o'] = (np.random.uniform(0.,1., n_heads*1) - 0.5) * 0.
        params['W_h_g_o'] = np.random.uniform(0.,1., (n_heads*1, hidden_dim)) - 0.5
        params['b_h_g_o'] = (np.random.uniform(0.,1., n_heads*1) - 0.5) * 0.
        params['W_h_s_o'] = np.random.uniform(0.,1., (n_heads*1, hidden_dim)) - 0.5
        params['b_h_s_o'] = (np.random.uniform(0.,1., n_heads*1) - 0.5) * 0.
        params['W_h_gamma_o'] = np.random.uniform(0.,1., (n_heads*1, hidden_dim)) - 0.5
        params['b_h_gamma_o'] = (np.random.uniform(0.,1., n_heads*1) - 0.5) * 0.
        params['W_h_er_o'] = np.random.uniform(0.,1., (n_heads*mem_shape[1], hidden_dim)) - 0.5
        params['b_h_er_o'] = (np.random.uniform(0.,1., n_heads*mem_shape[1]) - 0.5) * 0.
        params['W_h_a_o'] = np.random.uniform(0.,1., (n_heads*mem_shape[1], hidden_dim)) - 0.5
        params['b_h_a_o'] = (np.random.uniform(0.,1., n_heads*mem_shape[1]) - 0.5) * 0.
        
        tparams = OrderedDict()
        tparams['b_m'] = shared(name='b_m', value=params['b_m'].astype(config.floatX))
        tparams['b_w_mat'] = shared(name='b_w_mat', value=params['b_w_mat'].astype(config.floatX))
        tparams['W_e_i_h'] = shared(name='W_e_i_h', value=params['W_e_i_h'].astype(config.floatX))
        tparams['b_e_i_h'] = shared(name='b_e_i_h', value=params['b_e_i_h'].astype(config.floatX))
        tparams['W_r_h'] = shared(name='W_r_h', value=params['W_r_h'].astype(config.floatX))
        tparams['b_r_h'] = shared(name='b_r_h', value=params['b_r_h'].astype(config.floatX))
        tparams['W_h_e_o'] = shared(name='W_h_e_o', value=params['W_h_e_o'].astype(config.floatX))
        tparams['b_h_e_o'] = shared(name='b_h_e_o', value=params['b_h_e_o'].astype(config.floatX))
        tparams['W_h_k_o'] = shared(name='W_h_k_o', value=params['W_h_k_o'].astype(config.floatX))
        tparams['b_h_k_o'] = shared(name='b_h_k_o', value=params['b_h_k_o'].astype(config.floatX))
        tparams['W_h_beta_o'] = shared(name='W_h_beta_o', value=params['W_h_beta_o'].astype(config.floatX))
        tparams['b_h_beta_o'] = shared(name='b_h_beta_o', value=params['b_h_beta_o'].astype(config.floatX))
        tparams['W_h_g_o'] = shared(name='W_h_g_o', value=params['W_h_g_o'].astype(config.floatX))
        tparams['b_h_g_o'] = shared(name='b_h_g_o', value=params['b_h_g_o'].astype(config.floatX))
        tparams['W_h_s_o'] = shared(name='W_h_s_o', value=params['W_h_s_o'].astype(config.floatX))
        tparams['b_h_s_o'] = shared(name='b_h_s_o', value=params['b_h_s_o'].astype(config.floatX))
        tparams['W_h_gamma_o'] = shared(name='W_h_gamma_o', value=params['W_h_gamma_o'].astype(config.floatX))
        tparams['b_h_gamma_o'] = shared(name='b_h_gamma_o', value=params['b_h_gamma_o'].astype(config.floatX))
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
        b_m = self.tparams['b_m']
        b_w_mat = self.tparams['b_w_mat']
        W_e_i_h = self.tparams['W_e_i_h']
        b_e_i_h = self.tparams['b_e_i_h']
        W_r_h = self.tparams['W_r_h']
        b_r_h = self.tparams['b_r_h']
        W_h_e_o = self.tparams['W_h_e_o']
        b_h_e_o = self.tparams['b_h_e_o']
        W_h_k_o = self.tparams['W_h_k_o']
        b_h_k_o = self.tparams['b_h_k_o']
        W_h_beta_o = self.tparams['W_h_beta_o']
        b_h_beta_o = self.tparams['b_h_beta_o']
        W_h_g_o = self.tparams['W_h_g_o']
        b_h_g_o = self.tparams['b_h_g_o']
        W_h_s_o = self.tparams['W_h_s_o']
        b_h_s_o = self.tparams['b_h_s_o']
        W_h_gamma_o = self.tparams['W_h_gamma_o']
        b_h_gamma_o = self.tparams['b_h_gamma_o']
        W_h_er_o = self.tparams['W_h_er_o']
        b_h_er_o = self.tparams['b_h_er_o']
        W_h_a_o = self.tparams['W_h_a_o']
        b_h_a_o = self.tparams['b_h_a_o']
        
        mem_shape = self.mem_shape
        ext_out_dim = self.ext_out_dim
        n_heads = self.n_heads
        
        X = T.matrix('X')
        Y = T.matrix('Y')
        
        def _b_w_mat_update(b_w_t, k_t, beta_t, g_t, s_t, gamma_t, b_m):           
            w_t_c = (CAS(b_m, k_t, beta_t)).flatten()
            w_t_g = (Interpolate(w_t_c, b_w_t, g_t)).flatten()
            w_t_s = (ShiftConv(w_t_g, s_t, mem_shape[0])).flatten()
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
        
        def _step(X_t, b_m_t_1, b_w_mat_t_1, W_e_i_h_, b_e_i_h_, W_r_h_, b_r_h_, W_h_e_o_, b_h_e_o_, 
                    W_h_k_o_, b_h_k_o_, W_h_beta_o_, b_h_beta_o_, W_h_g_o_, b_h_g_o_,
                    W_h_s_o_, b_h_s_o_, W_h_gamma_o_, b_h_gamma_o_, W_h_er_o_, b_h_er_o_,
                    W_h_a_o_, b_h_a_o_):
            r_t_mat = ReadVec(b_m_t_1, b_w_mat_t_1)
            r_t = r_t_mat.flatten()
            h_t = T.nnet.sigmoid(W_e_i_h_.dot(X_t)+b_e_i_h_ + W_r_h_.dot(r_t)+b_r_h_)
            e_o_t = T.nnet.sigmoid(W_h_e_o_.dot(h_t)+b_h_e_o_)
            
            k_o_t = (W_h_k_o_.dot(h_t)+b_h_k_o_).reshape((n_heads, mem_shape[1]))
            #beta_o_t = (T.exp(W_h_beta_o_.dot(h_t)+b_h_beta_o_)).flatten()
            beta_o_t = (T.nnet.softplus(W_h_beta_o_.dot(h_t)+b_h_beta_o_)).flatten()
            g_o_t = (T.nnet.sigmoid(W_h_g_o_.dot(h_t)+b_h_g_o_)).flatten()
            s_o_t = (T.nnet.sigmoid(W_h_s_o_.dot(h_t)+b_h_s_o_)).flatten()
            #gamma_o_t = (T.log(T.exp(W_h_gamma_o_.dot(h_t)+b_h_gamma_o_)+1.)+1.).flatten()
            gamma_o_t = (T.nnet.softplus(W_h_gamma_o_.dot(h_t)+b_h_gamma_o_)+1.).flatten()
            
            er_o_t = (T.nnet.sigmoid(W_h_er_o_.dot(h_t)+b_h_er_o_)).reshape((n_heads, mem_shape[1]))
            #a_o_t = (T.nnet.sigmoid(W_h_a_o_.dot(h_t)+b_h_a_o_)).reshape((n_heads, mem_shape[1]))
            a_o_t = (W_h_a_o_.dot(h_t)+b_h_a_o_).reshape((n_heads, mem_shape[1]))
            
            w_t_f_mat, updates_w_t = theano.scan(
                _b_w_mat_update,
                non_sequences = [b_m_t_1],
                sequences = [b_w_mat_t_1, k_o_t, beta_o_t, g_o_t, s_o_t, gamma_o_t],
                outputs_info=None
            )
            
            b_m_t_seq, updates_m_t = theano.scan(
                _b_m_update,
                non_sequences = b_m_t_1,
                sequences = [w_t_f_mat, er_o_t, a_o_t],
                outputs_info=dict(initial=b_m_t_1)
            )
            
            b_w_mat_t = w_t_f_mat
            b_m_t = b_m_t_seq[-1]
            
            return [e_o_t, b_m_t, b_w_mat_t, r_t_mat, a_o_t]
        
        b_w_mat_soft = T.nnet.softmax(b_w_mat)
        
        [external_output, b_m_seq, b_w_mat_seq, debug_out_1, debug_out_2], all_updates = theano.scan(
            _step,
            sequences=[X],
            non_sequences=[W_e_i_h, b_e_i_h, W_r_h, b_r_h, W_h_e_o, b_h_e_o, 
                    W_h_k_o, b_h_k_o, W_h_beta_o, b_h_beta_o, W_h_g_o, b_h_g_o,
                    W_h_s_o, b_h_s_o, W_h_gamma_o, b_h_gamma_o, W_h_er_o, b_h_er_o,
                    W_h_a_o, b_h_a_o],
            outputs_info=[None, dict(initial=b_m), dict(initial=b_w_mat_soft), None, None]
        )
        
        o_err = T.mean(T.nnet.binary_crossentropy(external_output, Y))
        hamm_err = T.sum(T.neq(T.ge(external_output, 0.5)*1.0, Y))
        thresh_prediction = T.ge(external_output, 0.5)
        
        dtparams = T.grad(o_err, self.all_tparams)
        grad_abs_sum = T.sum(0)
        for g in dtparams:
            grad_abs_sum += T.sum(T.abs_(g))
        
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar("momentum")
        rmsprop_updates = self.opt.updates(self.all_tparams, dtparams, learning_rate, momentum)

        self.prediction = function([X], [external_output, thresh_prediction])
        self.prediction_plus_err = function([X, Y], [external_output, thresh_prediction, 
                                                    o_err, hamm_err, debug_out_1, debug_out_2, b_w_mat_seq])
        self.loss = function([X, Y], [o_err, hamm_err])
        self.rmsprop_step = function(
            [X, Y, learning_rate, momentum],
            [o_err, hamm_err],
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
    my_info['weights'] = info[6]
    np.savez(outfile, info=my_info)
    print "Saved info to %s." % outfile

def load_model_parameters(path, model):
    npzfile = np.load(path)
    model_params = npzfile['model_params'].tolist()
    model.load_model_params(model_params)
    print "Loaded model parameters from %s." % (path)

np.random.seed(2)
task_name = "COPY"
ext_in_dim = 8+2
ext_out_dim = 8
mem_shape = (128, 20)
n_heads = 1
hidden_dim = 100
model = NTM(ext_in_dim, ext_out_dim, mem_shape, n_heads, hidden_dim)
max_iter = 100000
max_len = 20
min_len = 1
learning_rate = 1e-3
momentum = 0.5
printFreq = 100
max_iter = 1000000
to_test = True
test_path = "../model/ntm-COPY-20-2016-03-03-21-40-59.npz"

if to_test:
    load_model_parameters(test_path, model)

time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
deb_f = None
if not to_test:
    debug_output_file = ("../plots/ntm_learning_curve-%s-%d-%s.txt" % (task_name, max_len, time_now))
    deb_f = open(debug_output_file, "w")
else:
    min_len = 10
    max_len = 120
    max_iter = 20

for iter in range(max_iter):
    # Generate input
    mylen = np.random.randint(min_len, max_len+1)
    X = np.zeros((2*mylen+2, ext_in_dim))
    Y = np.zeros((2*mylen+2, ext_out_dim))
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
                save_model_parameters("../model/ntm-%s-%d-%s.npz" % (task_name, max_len, time_now), model)
    else:
        info = model.prediction_plus_err(X, Y)
        print "Iter", iter, "Len:", mylen, "Error: ", info[2], info[3]
        time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_info("../plots/ntm-info-%s-%d-%d-%d-%s.npz" % (task_name, min_len, max_len, mylen, time_now), X, Y, info)
        print_info("../plots/ntm-info-%s-%d-%d-%d-%s.npz" % (task_name, min_len, max_len, mylen, time_now), 
                                    ext_in_dim, ext_out_dim, mem_shape, task_name)

if not to_test:
    deb_f.close()
    print_learning_curve(debug_output_file, 20, task_name)