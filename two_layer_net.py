from layers import *
from builtins import range
from builtins import object
import time

class TwoLayerFCNet(object):
    """
    模块化的两层全连接网络，使用ReLU作为激活函数，softmax作为损失。
    网络结构为affine - relu - affine - softmax
    """
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3,reg=1e-3):
        """
        - weight_scale: 初始化权重时的标准差
        """
        self.params = {}
        self.reg = reg
        
        # 初始化权重和偏置
        self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['W2'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b1'] = np.zeros((hidden_dim))
        self.params['b2'] = np.zeros((num_classes))
        
    # def loss(self, X, y=None):
    #     """
    #     输入:
    #     - X: 维度为(N, d_1, ..., d_k)
    #     - y: 维度为(N,)
    #     输出:
    #     如果y==None返回scores
    #     - scores: 维度为(N, C)
    #     否则，返回：
    #     - loss: 损失
    #     - grads: self.params里的参数的梯度
    #     """
    #     scores = None
        
    #     # 实现前向传播，计算得分
    #     W1,b1 = self.params['W1'],self.params['b1']
    #     W2,b2 = self.params['W2'],self.params['b2']
    #     out_relu,affine_relu_cache = affine_relu_forward(X, W1, b1)
    #     scores,fc_cache = affine_forward(out_relu,W2,b2)
        
    #     # 如果y==None，返回scores
    #     if y is None:
    #         return scores
        
    #     loss, grads = 0, {}
    #     # dscores维度(N,C)
    #     loss,dscores = softmax_loss(scores, y)
    #     # 加上正则化，loss/N在softmax_loss里已经实现了
    #     loss += self.reg*0.5*(np.sum(W1**2)+np.sum(W2**2))
        
    #     # 求梯度
    #     dout_relu,dW2,db2 = affine_backward(dscores,fc_cache)
    #     dx,dW1,db1 = affine_relu_backward(dout_relu,affine_relu_cache)
        
    #     grads['W1'] = dW1+self.reg*W1
    #     grads['b1'] = db1
    #     grads['W2'] = dW2+self.reg*W2
    #     grads['b2'] = db2
        
    #     return loss, grads
    
    def loss(self,X,Y=None):
        '''
        计算损失函数
        '''
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        ##############################
        #Computing the class scores of the input
        ##############################
        Z1 = X.dot(W1) + b1#第一层
        S1 = np.maximum(0,Z1)#隐藏层激活函数
        # sigmoid activation
        # S1 = 1 / (1 + np.exp(-Z1))
        # leaky relu
        # S1 = np.maximum(Z1,0.01*Z1)
        score = S1.dot(W2) + b2#输出层
 
        if Y is None:
            return score
        loss = None
        ###############################
        #TODO:forward pass
        #computing the loss of the net
        ################################
        exp_scores = np.exp(score)
        probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
        #数据损失
        data_loss = -1.0/ N * np.log(probs[np.arange(N),Y]).sum()
        #正则损失
        reg_loss = 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))
        #总损失
        loss = data_loss + reg_loss
        ################################
        #TODO:backward pass
        #computing the gradient
        ################################
        grads = {}
        dscores = probs
        dscores[np.arange(N),Y] -= 1
        dscores /= N
        #更新W2B2
        grads['W2'] = S1.T.dot(dscores) + self.reg *W2
        grads['b2'] = np.sum(dscores,axis = 0)
 
        #第二层
 
        dhidden = dscores.dot(W2.T)
        dhidden[S1<=0] = 0
 
        grads['W1'] = X.T.dot(dhidden) + self.reg *W1
        grads['b1'] = np.sum(dhidden,axis = 0)
 
        return loss,grads
        
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        self.reg = reg
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        #print("iterations_per_epoch",num_train,batch_size,num_train / batch_size,iterations_per_epoch)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            index = np.random.choice(num_train, batch_size, replace=num_train < batch_size)
            X_batch = X[index]
            y_batch = y[index]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, Y=y_batch)
            loss_history.append(loss)

            #########################################################################
            # Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
            # Use the gradients to update the parameters of the network
            
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]
        
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % batch_size == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                #print("val_acc_history",val_acc_history)
                #print("train_acc_history",train_acc_history)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        y_pred = np.argmax(self.loss(X), axis=1) 
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred