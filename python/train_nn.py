import numpy as np
import scipy.io
from nn import *
import matplotlib
import matplotlib.pyplot as plt
import string
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')


train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
# Smaller batches: (+) better train accuracy, (-) slows down execution, (?) valid accuracy
batch_size = 120  # 90 batches for 10800 samples
learning_rate = 5e-3  # 5e-2 oscillates out of control, and 5e-4 learns too slow
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, 64, params, 'layer1');
initialize_weights(64, 36, params, 'output')

orig_backend = matplotlib.get_backend()
matplotlib.use('agg')
weightsToPlot = params["Wlayer1"]
weightsToPlot = weightsToPlot.reshape(32, 32, 64)
fig = plt.figure(figsize=(8., 8.))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)
for ax, im in zip(grid, [weightsToPlot[:, :, i] for i in range(64)]):
    ax.imshow(im)

plt.savefig("untrained_weights.png")
matplotlib.use(orig_backend)

# with default settings, should get accuracy > 80%
epoch_list = []
train_acc_list = []
train_loss_list = []
valid_acc_list = []
valid_loss_list = []

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        y_idx = np.argmax(yb, axis=1)  # Convert from one hot to numerical labels

        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs
        delta1[np.arange(probs.shape[0]), y_idx] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        for k, v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name] = params[name] - learning_rate * params[k]

    if itr % 2 == 0:  # Report results
        train_loss = total_loss / len(batches)
        train_acc = total_acc / len(batches)

        # run on validation set and report accuracy! should be above 75%
        # forward
        h1 = forward(valid_x, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # Compute loss and accuracy
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
        print(f"Epoch {itr}. Train acc: {round(train_acc, 2)}. Train loss: {round(train_loss, 2)}. "
              f"Valid acc: {round(valid_acc, 2)}. Valid loss: {round(valid_loss, 2)}")

        # Add to lists to plot later
        epoch_list.append(itr)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

plt_performance = False
if plt_performance:
    fig, (acc_plt, loss_plt) = plt.subplots(2, 1)
    fig.suptitle(f"Learning rate = {learning_rate}. Batch size = {batch_size}")
    acc_plt.plot(epoch_list, train_acc_list, label="Training accuracy")
    acc_plt.plot(epoch_list, valid_acc_list, label="Validation accuracy")
    acc_plt.set_ylabel("Accuracy")
    acc_plt.legend()
    loss_plt.plot(epoch_list, train_loss_list, label="Training loss")
    loss_plt.plot(epoch_list, valid_loss_list, label="Validation loss")
    loss_plt.set_ylabel("Loss")
    loss_plt.legend()
    plt.show()

# Accuacy on test set
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
# Compute loss and accuracy
test_loss, test_acc = compute_loss_and_acc(test_y, probs)
print(f"Final test acc: {round(test_acc, 2)}.")

view_data = False
if view_data: # view the data
    for crop in xb:
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

save_pickle = True
if save_pickle:
    saved_params = {k:v for k,v in params.items() if '_' not in k}
    with open('nn_weights.pickle', 'wb') as handle:
        pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

orig_backend = matplotlib.get_backend()
matplotlib.use('agg')
weightsToPlot = params["Wlayer1"]
weightsToPlot = weightsToPlot.reshape(32, 32, 64)
fig = plt.figure(figsize=(8., 8.))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)
for ax, im in zip(grid, [weightsToPlot[:, :, i] for i in range(64)]):
    ax.imshow(im)

plt.savefig("trained_weights.png")
matplotlib.use(orig_backend)

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
# forward
h1 = forward(train_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
y_idx = np.argmax(train_y, axis=1)
prob_idx = np.argmax(probs, axis=1)
for (x, y) in zip(prob_idx, y_idx):
    confusion_matrix[x, y] += 1


plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
