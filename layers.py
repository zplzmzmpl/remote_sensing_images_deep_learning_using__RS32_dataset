import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x = x.reshape(x.shape[0],-1)
    out = x.dot(w)+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx=np.dot(dout,w.T)
    dx=np.reshape(dx,x.shape)
    x_new=x.reshape(x.shape[0],-1)
    dw=np.dot(x_new.T,dout) 
    db=np.sum(dout,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x >= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


#---------------选做----------------
def leaky_relu_forward(x, alpha=0.01):
    out = np.maximum(x, alpha * x)
    cache = x
    return out, cache

def leaky_relu_backward(dout, cache, alpha=0.01):
    x = cache
    dx = np.where(x > 0, dout, alpha * dout)
    return dx

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    x = cache
    sigmoid = 1 / (1 + np.exp(-x))
    dx = dout * sigmoid * (1 - sigmoid)
    return dx

#---------------选做----------------

def svm_loss(x, y):
    """
    输入:
    - x: 维度为(N, C)，其中x[i, j]表示第i个输入关于第j个类的得分
    - y: 标签，维度为(N,)
    返回:
    - loss: 损失
    - dx: x（也就是得分）的梯度
    """
    N = x.shape[0]
    # x[np.arange(N), y]得到的维度是(N,)
    correct_class_scores = x[np.arange(N), y]
    # correct_class_scores[:, np.newaxis]的维度是(N,1)
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    # 正确类得分不参与计算
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    
    # 计算梯度
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    # 减去最大值，使其全部小于0，这样exp(shifted_logits)就会在0到1之间，
    # 防止exp(x)过大而溢出。而且该减去操作不影响最终损失。
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    N = x.shape[0]
    # 只使用正确类的得分
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    # 计算梯度
    probs = np.exp(log_probs)
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


