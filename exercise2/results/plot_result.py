import matplotlib.pyplot as plt
import json

data = []
plt.figure(1)
for i in [0, 1, 2, 3]:
    path = 'results_run_' + str(i) + '.json'
    with open(path) as f:
        data.append(json.load(f))
        label = 'Learning Rate = ' + str(data[-1]['lr'])
        plt.plot(data[-1]['learning_curve'], label=label)
plt.legend(loc='right')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.ylim(top=2.5)
plt.savefig('lr.png')


plt.figure(2)
for i in [4, 5, 6, 7]:
    path = 'results_run_' + str(i) + '.json'
    with open(path) as f:
        data.append(json.load(f))
        label = 'Filter Size = ' + str(data[-1]['filter_size'])
        plt.plot(data[-1]['learning_curve'], label=label)
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.ylim(top=2.5)
plt.savefig('filter_size.png')

