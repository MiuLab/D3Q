__author__ = 'pberlin'
# from .utils import *
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams,ignore=None):
    l = []
    print 'start new', ignore
    for kk, vv in tparams.iteritems():
        if ignore != None and ignore in kk:
            continue
        print kk
        l.append(vv)
    return l


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def init_tparams_value(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        print kk
        if kk == 'hout':
            pass
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.rand(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)

def sigmoid(x):
    return tensor.nnet.tanh(x)

def linear(x):
    return x

def param_init_fflayer(params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def param_init_gru(params, prefix='gru', nin=None, dim=None):


    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, prefix='gru', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                strict=True)
    # rval = [rval]
    return rval

def fflayer(tparams, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])

def rmsprop(tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    # f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, allow_input_downcast=True)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    # f_update = theano.function([], [], updates=updir_new+param_up,
    #                            on_unused_input='ignore')

    return updir_new+param_up

def RMSprop(cost, params, lr=0.0001, rho=0.999, epsilon=1e-8):
    grads = tensor.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = tensor.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def adam(cost, tparams):

    grads = theano.grad(cost,wrt=itemlist(tparams))

    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]


    lr0 = 0.0025
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    return gsup+updates

def clip_grad(grads, clip_c):

    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    return grads

numpy.random.seed(2017)

class ModelBasedUsersimulator:
    def __init__(self, input_size, hidden_size, action_size):
        self.model = {}

        params = OrderedDict()
        param_init_gru(params, 'gru', input_size, hidden_size)
        param_init_fflayer(params,'h_diaact',hidden_size, action_size)

        self.n_diaact = action_size
        self.param = params
        self.tparams = init_tparams(params)

        self.gamma = 0.9
        self.reg_l2 = 1e-3

        s_t = tensor.tensor3('s_t', dtype='float32')
        diaact_t = tensor.matrix('diaact_t', dtype='float32')


        h = gru_layer(self.tparams, s_t, 'gru')

        pre_diaact = fflayer(self.tparams, h, 'h_diaact', activ='linear')
        pre_diaact = pre_diaact.reshape([pre_diaact.shape[0] * pre_diaact.shape[1], pre_diaact.shape[2]])
        prob_diaact = theano.tensor.nnet.softmax(pre_diaact)

        diaact_loss = -diaact_t * tensor.log(prob_diaact)
        diaact_loss = diaact_loss.mean()


        total_cost = diaact_loss

        updates = RMSprop(total_cost,itemlist(self.tparams))

        self.train = theano.function(inputs=[s_t, diaact_t], outputs=total_cost , allow_input_downcast=True,
                                     on_unused_input='ignore', updates=updates)

        rng = theano.sandbox.rng_mrg.MRG_RandomStreams()
        sample_diaact = rng.multinomial(n=5, pvals=prob_diaact)

        self.train_sample = theano.function(inputs=[s_t, diaact_t], outputs=sample_diaact, allow_input_downcast=True,
                                     on_unused_input='ignore')

        self.sample_fn = self.build_sampler()

        # pre_slots = fflayer(self.tparams,h,'hout',activ='sigmoid')
        #
        # h_shape = h.shape
        #
        # tensor.clip(prob, 1e-5, 1-1e-5)
        # a_t_flat = a_t.flatten()
        # a_t_flat_idx = tensor.arange(a_t_flat.shape[0]) * self.n_action + a_t_flat
        #
        #
        # cost = tensor.log(prob.flatten()[a_t_flat_idx]) * r.flatten()
        # # cost = tensor.log(prob.flatten()[a_t_flat_idx])# * r.flatten() * m_t.flatten()
        # # cost = cost.reshape([a_t.shape[0],a_t.shape[1]])
        # # cost = cost * m_t * r
        #
        # cost_sup = -tensor.log(prob.flatten()[a_t_flat_idx]).mean()
        #
        # entropy_cost = -prob * tensor.log(prob)
        # # entropy_cost = tensor.sum(entropy_cost )
        # #
        # # weights = [(i**2).sum() for i in itemlist(self.tparams)]
        # total_cost = -cost.sum() / s_t.shape[0] - 0.1 * entropy_cost.sum() / s_t.shape[0]#+ 1e-4 * tensor.sum(weights)# + 0.01 * entropy_cost / s_t.shape[0]
        # #
        # prediction = tensor.argmax(prob,axis=1)


        # updates = RMSprop(total_cost,itemlist(self.tparams, 'h_value'))
        # update_sup = RMSprop(cost_sup,itemlist(self.tparams,'h_value'))
        # update_value = RMSprop(value_loss,itemlist(self.tparams,'hout'))
        # updates = adam(total_cost,self.tparams)

        # self.train = theano.function(inputs=[s_t, a_t, r], outputs=total_cost, allow_input_downcast=True,  on_unused_input='ignore', updates=updates)
        # self.train_value = theano.function(inputs=[s_t, r], outputs=value_loss, allow_input_downcast=True,  on_unused_input='ignore', updates=update_value)
        # self.predict_value = theano.function(inputs=[s_t], outputs=h_value, allow_input_downcast=True)


        # self.train_batch = theano.function(inputs=[s_t, a_t, m_t, r], outputs=total_cost, allow_input_downcast=True,  on_unused_input='ignore', updates=updates)

        # self.predict = theano.function(inputs=[s_t], outputs=prediction, allow_input_downcast=True,  on_unused_input='ignore')

        # self.predict_debug = theano.function(inputs=[s_t], outputs=prob, allow_input_downcast=True,  on_unused_input='ignore')

        # self.sampler = self.build_sampler()

        # rng = theano.sandbox.rng_mrg.MRG_RandomStreams()
        #
        # self.sampler = theano.function(inputs=[s_t], outputs=tensor.argmax(rng.multinomial(n=1, pvals=prob)),allow_input_downcast=True,  on_unused_input='ignore')

        # self.train_sup = theano.function(inputs=[s_t, a_t], outputs=cost_sup, allow_input_downcast=True,  on_unused_input='ignore', updates=update_sup)

    def build_sampler(self):
        s_t = tensor.matrix('s_t', dtype='float32')
        h_last = tensor.matrix('h_tm1', dtype='float32')

        state_below = s_t

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        prefix = 'gru'
        dim = self.tparams[_p(prefix, 'Ux')].shape[1]

        # utility function to slice a tensor
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        state_below_ = tensor.dot(state_below, self.tparams[_p(prefix, 'W')]) + \
            self.tparams[_p(prefix, 'b')]
        # input to compute the hidden state proposal
        state_belowx = tensor.dot(state_below, self.tparams[_p(prefix, 'Wx')]) + \
            self.tparams[_p(prefix, 'bx')]

        # step function to be used by scan
        # arguments    | sequences |outputs-info| non-seqs
        def _step_slice(x_, xx_, h_, U, Ux):
            preact = tensor.dot(h_, U)
            preact += x_

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_

            # hidden state proposal
            h = tensor.tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_ + (1. - u) * h
            # h = m_[:, None] * h + (1. - m_)[:, None] * h_
            return h

        # prepare scan arguments


        _step = _step_slice
        shared_vars = [self.tparams[_p(prefix, 'U')],
                       self.tparams[_p(prefix, 'Ux')]]

        h = _step_slice(state_below_, state_belowx, h_last, *shared_vars)

        pre_diaact = fflayer(self.tparams, h, 'h_diaact', activ='linear')
        prob_diaact = tensor.nnet.softmax(pre_diaact)

        pre_slots = fflayer(self.tparams, h, 'h_slots', activ='linear')
        prob_slots = theano.tensor.nnet.sigmoid(pre_slots)

        rng = theano.sandbox.rng_mrg.MRG_RandomStreams()

        sampler_fn = theano.function(inputs=[s_t, h_last], outputs=[tensor.argmax(rng.multinomial(n=2, pvals=prob_diaact)), prob_slots,h],allow_input_downcast=True,  on_unused_input='ignore')

        return sampler_fn


    def singleBatch(self,inputs):
        out = {}
        out['cost'] = {'total_cost':self.train(*inputs)}
        # self.f_update()
        return out

    def reset_hat(self):
        tp, tp_hat = self.tparams,self.tparams_hat
        for i in tp.items():
            k,v = i
            tp_hat[k].set_value(v.get_value())

    def unzip(self):
        return unzip(self.tparams)

    def load(self, path):
        pp = numpy.load(path)
        for kk,vv in self.param.items():
            if kk not in pp:
                continue
            self.param[kk] = pp[kk]
        self.tparams=init_tparams(self.param)

# class DQN:
#
#     def __init__(self, input_size, hidden_size, output_size):
#         self.model = {}
#
#
#         params = OrderedDict()
#
#         params['Wxh'] = norm_weight(input_size, hidden_size, scale=0.05)
#         params['bxh'] = numpy.zeros((hidden_size,)).astype('float32')
#
#         params['Wd'] = norm_weight(hidden_size, output_size, scale=0.01)
#         params['bd'] = numpy.zeros((output_size,)).astype('float32')
#
#         params_hat = OrderedDict()
#         params_hat['Wxh'] = norm_weight(input_size, hidden_size, scale=0.05)
#         params_hat['bxh'] = numpy.zeros((hidden_size,)).astype('float32')
#
#         params_hat['Wd'] = norm_weight(hidden_size, output_size, scale=0.01)
#         params_hat['bd'] = numpy.zeros((output_size,)).astype('float32')
#
#         self.param = params
#         self.tparams = init_tparams(params)
#         self.tparams_hat = init_tparams(params_hat)
#
#         self.gamma = 0.9
#         self.reg_l2 = 1e-3
#
#         s_t = tensor.matrix('s_t', dtype='float32')
#         s_t_predict = tensor.matrix('s_t_predict', dtype='float32')
#         s_tp1 = tensor.matrix('s_tp1', dtype='float32')
#         a_idx = tensor.matrix('a_idx', dtype='int32')
#         r = tensor.matrix('r', dtype='float32')
#         term = tensor.matrix('term', dtype='int32')
#
#         def fwdPass(x):
#             h = tensor.dot(x, self.tparams['Wxh']) + self.tparams['bxh']
#             h = tensor.nnet.sigmoid(h)
#             h = tensor.dot(h, self.tparams['Wd']) + self.tparams['bd']
#             return h
#
#         def fwdPass_hat(x):
#             h = tensor.dot(x, self.tparams_hat['Wxh']) + self.tparams_hat['bxh']
#             h = tensor.nnet.relu(h)
#             h = tensor.dot(h, self.tparams_hat['Wd']) + self.tparams_hat['bd']
#             return h
#
#
#         self.reset_hat()
#         q_v = fwdPass(s_t)
#         q_v_tp1 = fwdPass(s_tp1)
#         q_v_tp1 = theano.gradient.disconnected_grad(q_v_tp1)
#         q_v_predict = fwdPass(s_t_predict)
#
#
#         a_idx_flat = a_idx[:,0]
#         term_flat = term[:,0]
#         r_flat = r[:,0]
#         diff = q_v[theano.tensor.arange(0,a_idx.shape[0]), a_idx_flat] - ( r_flat + (1-term_flat) * self.gamma * tensor.max(q_v_tp1, axis=1, keepdims=True)[:,0])
#         diff = 0.5 * (diff ** 2)
#         diff = theano.tensor.sum(diff) #+ self.reg_l2 * (theano.tensor.sum(self.tparams['Wxh'] ** 2) + theano.tensor.sum(self.tparams['Wd'] ** 2) )
#
#         total_cost = diff / s_t.shape[0]
#
#
#
#         # grads = theano.grad(total_cost,wrt=itemlist(self.tparams))
#
#         updates = RMSprop(total_cost,itemlist(self.tparams))
#
#         # grads = clip_grad(grads,1)
#
#         # self.f_grad_shared, self.f_update = sgd(self.tparams,grads,[s_t, a_idx, r, s_tp1, term],total_cost)
#         self.train = theano.function(inputs=[s_t, a_idx, r, s_tp1, term], outputs=total_cost, allow_input_downcast=True,  on_unused_input='ignore', updates=updates)
#         self.debug = theano.function(inputs=[s_t, a_idx, r, s_tp1, term], outputs=[q_v[theano.tensor.arange(0,a_idx.shape[0]), a_idx_flat], q_v, a_idx], allow_input_downcast=True,on_unused_input='ignore')
#         self.predict = theano.function(inputs=[s_t_predict], outputs=tensor.argmax(q_v_predict,axis=1), allow_input_downcast=True)
#         self.predict_value = theano.function(inputs=[s_t_predict], outputs=q_v_predict, allow_input_downcast=True)
#
#     def singleBatch(self,inputs):
#         out = {}
#         out['cost'] = {'total_cost':self.train(*inputs)}
#         # self.f_update()
#         return out
#
#     def reset_hat(self):
#         tp, tp_hat = self.tparams,self.tparams_hat
#         for i in tp.items():
#             k,v = i
#             tp_hat[k].set_value(v.get_value())
#
#     def unzip(self):
#         return unzip(self.tparams)
#
#     def get_norm(self):
#         tp = self.tparams
#         vv = 0
#         for i in tp.items():
#             k,v = i
#             vv  += v.get_value().sum()
#         return vv
#
#     def load(self, path):
#         pp = numpy.load(path)
#         for kk,vv in self.param.items():
#             if kk not in pp:
#                 continue
#             self.param[kk] = pp[kk]
#         self.tparams=init_tparams(self.param)
