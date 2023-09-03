import emnist
import json
import numpy as np
from common.conv_net import SimpleConvNet
from common.trainer import Trainer

# Load local emnist mapping json

with open('emnist_balanced_mapping.json', 'r') as f:
    emnist_mapping = json.load(f)

(X_train, y_train) = emnist.extract_training_samples('balanced')
(X_test, y_test) = emnist.extract_test_samples('balanced')

# Print the unique label values
unique_labels = set(y_train)


# Make x_train and x_test into 4 dimensional arrays (60000, 1, 28, 28)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

max_epochs = 10

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=len(unique_labels), weight_init_std=0.01)

trainer = Trainer(network, X_train, y_train, X_test, y_test,
                    epochs=max_epochs, mini_batch_size=100,
                    optimizer='Adam', optimizer_param={'lr': 0.001},
                    evaluate_sample_num_per_epoch=1000)

trainer.train()
network.save_params("emnist_params.pkl")
print("Saved Network Parameters!")

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()