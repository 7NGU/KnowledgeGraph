import networkx as nx
import numpy as np
import csv
from node2vec import Node2Vec
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

my_matrix = np.loadtxt(open("Facebook.csv"), dtype=int, delimiter=",", skiprows=0)
X, y = my_matrix[:, :-1], my_matrix[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train = np.column_stack((X_train, y_train))
np.savetxt('train_usual.csv', train, fmt="%d", delimiter=',')
test = np.column_stack((X_test, y_test))
np.savetxt('test_usual.csv', test, fmt="%d", delimiter=',')

node = []
edge_train = []
edge_test = []
edge_exit = []
count = 0
with open("Facebook.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        edge_exit.append(row)
        count += 1
        if row[0] not in node:
            node.append(row[0])
        if row[1] not in node:
            node.append(row[1])

with open("test_usual.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        edge_test.append(row)

with open("train_usual.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        edge_train.append(row)


# with open('facebook_combined.txt', 'r') as f:
#     data = f.readlines()
#     for line in data:
#         line = tuple(line.replace('\n', '').split(' '))
#         edge_exit.append(line)
#
#         if line[0] not in node:
#             node.append(line[0])
#
#         if line[1] not in node:
#             node.append(line[1])
#
#         if (count % 4 == 0):
#             edge_test.append(line)
#             count += 1
#         else:
#             edge_train.append(line)
#             count += 1

G = nx.DiGraph()
G.add_nodes_from(node)
G.add_edges_from(edge_train)

G_all = nx.DiGraph()
G_all.add_nodes_from(node)
G_all.add_edges_from(edge_exit)

edge_not_exit = list(nx.non_edges(G_all))

node2vec = Node2Vec(G, dimensions=128, walk_length=100, num_walks=100, workers=10)

model = node2vec.fit()  # trainning process

# len_not = len(edge_not_exit)
# len_test = len(edge_test)
# score_total = 0.0
# times = 0.0
# for i in range(len_test):
#     sim_test = model.wv.similarity(edge_test[i][0], edge_test[i][1])
#     j = 0
#     while (j < len_not):
#         sim_not = model.wv.similarity(edge_not_exit[j][0], edge_not_exit[j][1])
#         times += 1.0
#         j += 10000
#         if (sim_test > sim_not):
#             score_total += 1.0
#         elif (sim_test == sim_not):
#             score_total += 0.5
#         else:
#             score_total += 0.0
#
# #     for j in range(int(len_not/10000)):
# #         sim_not = model.wv.similarity(edge_not_exit[j][0], edge_not_exit[j][1])
# #         times += 1.0
# #         if(sim_test > sim_not):
# #             score_total += 1.0
# #         elif(sim_test == sim_not):
# #             score_total += 0.5
# #         else:
# #             score_total += 0.0
#
# score = score_total / times
# print(score)

len_not = len(edge_not_exit)
len_test = len(edge_test)
y_true = np.array([])
y_scores = np.array([])

for i in range(len_test):
    y_true = np.append(y_true, 1)
    sim = model.wv.similarity(edge_test[i][0], edge_test[i][1])
    y_scores = np.append(y_scores, sim)

j = 0
while (j < len_not):
    y_true = np.append(y_true, 0)
    sim = model.wv.similarity(edge_not_exit[i][0], edge_not_exit[i][1])
    y_scores = np.append(y_scores, sim)
    j += 100

print(roc_auc_score(y_true, y_scores))
