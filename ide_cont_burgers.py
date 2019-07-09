#%% IMPORTING/SETTING UP PATHS

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Manually making sure the numpy random seeds are "the same" on all devices, for reproducibility in random processes
np.random.seed(1234)
# Same for tensorflow
tf.random.set_seed(1234)

repoPath = 'PINNs'
utilsPath = os.path.join(repoPath, 'Utilities')
dataPath = os.path.join(repoPath, 'main', 'Data')
appDataPath = os.path.join(repoPath, 'appendix', 'Data')

sys.path.insert(0, utilsPath)
from plotting import newfig, savefig

#%% HYPER PARAMETERS
# Data size on the solution u
N_u = 2000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Creating the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
epochs = 10000
#%% PREPARING THE DATA

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

# Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
data = scipy.io.loadmat(os.path.join(appDataPath, 'burgers_shock.mat'))

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T

# Meshing x and t in 2D (256,100)
X, T = np.meshgrid(x,t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Preparing the testing u_star
u_star = Exact_u.flatten()[:,None]

# Domain bounds (lowerbounds upperbounds) [x, t], which are here ([-1.0, 0.0] and [1.0, 1.0])
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)
               
# # Getting the initial conditions (t=0)
# xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
# uu1 = Exact_u[0:1,:].T
# # Getting the lowest boundary conditions (x=-1) 
# xx2 = np.hstack((X[:,0:1], T[:,0:1]))
# uu2 = Exact_u[:,0:1]
# # Getting the highest boundary conditions (x=1) 
# xx3 = np.hstack((X[:,-1:], T[:,-1:]))
# uu3 = Exact_u[:,-1:]
# # Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
# X_u_train = np.vstack([xx1, xx2, xx3])
# u_train = np.vstack([uu1, uu2, uu3])

# # Generating the x and t collocation points for f, with each having a N_f size
# # We pointwise add and multiply to spread the LHS over the 2D domain
# X_f_train = lb + (ub-lb)*lhs(2, N_f)

# # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
# idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
# # Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
# X_u_train = X_u_train[idx,:]
# # Getting the corresponding u_train
# u_train = u_train [idx,:]

# Noiseless data
noise = 0.0            
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]

#%% CREATING THE MODEL

# Adding the different basic Keras layers (with glorot==Xavier init of weights)
u_model = tf.keras.Sequential()
u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
for w in layers[1:]:
  u_model.add(tf.keras.layers.Dense(w, activation=tf.nn.tanh,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'))
print(u_model.summary())

def f_model(x, t, lambda_1, lambda_2):
  # Using the new GradientTape paradigm of TF2.0,
  # which keeps track of operations to get the gradient at runtime
  with tf.GradientTape(persistent=True) as tape:
    # Watching the two inputs we’ll need later, x and t
    tape.watch(x)
    tape.watch(t)
    tape.watch(lambda_1)
    tape.watch(lambda_2)
    # Packing together the inputs
    X = tf.stack([x[:,0], t[:,0]], axis=1)
    # Getting the prediction
    u = u_model(X)
    # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
    u_x = tape.gradient(u, x)
    l1 = lambda_1
    l2 = tf.exp(lambda_2)
  
  # Getting the other derivatives
  u_xx = tape.gradient(u_x, x)
  u_t = tape.gradient(u, t)

  # Letting the tape go
  del tape

  # Buidling the PINNs
  return u_t + l1*u*u_x - l2*u_xx

def pinnLoss(u_model, x, t, lambda_1, lambda_2):
  # Defining custom loss to be returned
  def loss(u, u_pred):
    f_pred = f_model(x, t, lambda_1, lambda_2)
    return tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(f_pred)) + tf.reduce_mean(lambda_1*0 + lambda_2*0)
  
  return loss

# Setting up tensorboard
logdir = os.path.join("logs", "scalars", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Compiling and training the Keras model
x_f = tf.convert_to_tensor(X_u_train[:, 0:1], dtype="float32")
t_f = tf.convert_to_tensor(X_u_train[:, 1:2], dtype="float32")
lambda_1 = tf.Variable([0.0], dtype=tf.float32, trainable=True)
lambda_2 = tf.Variable([-6.0], dtype=tf.float32, trainable=True)
u_model.layers[-1].trainable_weights.extend([lambda_1, lambda_2])
u_model.compile(loss=pinnLoss(u_model, x_f, t_f, lambda_1, lambda_2), optimizer=optimizer)

#%% TRAINING THE MODEL

# Defining a custom logger
class CustomLogger(tf.keras.callbacks.Callback):
  def __init__(self, n, lambda_1, lambda_2):
    self.start_time = time.time()
    self.lambda_1 = lambda_1
    self.lambda_2 = lambda_2
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      loss = logs.get('loss')
      elapsed = datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")
      print(f"epoch = {epoch:6d}  elapsed = {elapsed}  loss = {loss:.4e}  " + \
        f"l1 = {self.lambda_1.numpy()}  l2 = {self.lambda_2.numpy()}")

# Doing the training
u_model.fit(X_u_train, u_train, batch_size=None, epochs=epochs,
  verbose=0, callbacks=[tensorboard_callback, CustomLogger(10, lambda_1, lambda_2)])

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred = u_model.predict(X_star)

f_pred = f_model(x_f, t_f)

# Getting the relative error for u
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))

# Interpolating the results on the whole (x,t) domain.
# griddata(points, values, points at which to interpolate, method)
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

#%% PLOTTING THE RESULTS

fig, ax = newfig(1.0, 1.1)
ax.axis('off')

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: u(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = 0.25$', fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 0.75$', fontsize = 10)

plt.show()
# savefig('./inf_cont_burgers')