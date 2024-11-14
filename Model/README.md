# Model
MUST call the deep_kernel_utils.py file, in order to transform inputs through a deep neural network before passing them through the kernel.

GP.py defines the deep kernel GP.
train.py defines the training loop.

An alternative (unsure if easier or harder) way to implement the GP model is to take all the code from here: (https://github.com/yifanc96/NonLinPDEs-GPsolver), and
then before passing in the data, transform it using the Deep_Transform network. The problem is that it might be difficult to pass gradients back to Deep_Transform parameters.