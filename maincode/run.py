import tensorflow as tf
import torch
from constants import LR, eta, epochs
from utils import load_data
from utils import convert_sparse_matrix_to_sparse_tensor
from trainer import Trainer
from re_run import total_data
import getopt, sys
import os
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "e:"
long_options = ["lr=", "lambda=", "eta=", "epochs="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    print (str(err))
    sys.exit(2)

for current_argument, current_value in arguments:
    if current_argument in ("--lr"):
        LR = float(current_value)
    elif current_argument in ("-e","--eta"):
        eta = float(current_value)
    elif current_argument in ("--epochs"):
        epochs = int(current_value)
    else:
        raise Exception(f"No such parameter named {current_argument}")
    




if __name__ == "__main__":

    path = "boat-sampling"
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    data = load_data(path,device=device)
    graph = data[0]
    labels = [x[1] for x in data[1]]
    n_clusters = data[2]
    adj = data[3]
    n = data[4]

    trainer = Trainer(device=device)
    trainer.initialize_data(adj, graph, eta, n_clusters, LR, labels, epochs, n)
    trainer.train()






