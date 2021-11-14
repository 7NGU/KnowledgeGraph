from tqdm import tqdm
from gensim.models import word2vec
import networkx as nx
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def walkOneTime(g, start_node, walk_length):
    walk = [str(start_node)]  # 初始化游走序列
    for _ in range(walk_length):  # 最大长度范围内进行采样
        current_node = int(walk[-1])
        if g.has_node(current_node):
            successors = list(g.successors(current_node)) # graph.successor: 获取当前节点的后继邻居
            if len(successors) > 0:
                next_node = np.random.choice(successors, 1)
                walk.extend([str(n) for n in next_node])
    return walk

def getDeepwalkSeqs(g, walk_length, num_walks):
    seqs=[]
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(list(g.node))
        w = walkOneTime(g,start_node, walk_length)
        seqs.append(w)
    return seqs

def deepwalk(g, dimensions = 10, walk_length = 80, num_walks = 10, min_count = 3 ):
    seqs = getDeepwalkSeqs(g, walk_length = walk_length, num_walks = num_walks)
    model = word2vec.Word2Vec(seqs, vector_size = dimensions, min_count = min_count)
    return model

if __name__ == '__main__':
    # g = nx.fast_gnp_random_graph(n = 100, p = 0.5,directed = True) #快速随机生成一个有向图
    # print(model.wv.most_similar('2', topn=3))  # 观察与节点2最相近的三个节点
    # model.wv.save_word2vec_format('e.emd') # 可以把emd储存下来以便下游任务使用
    # model.save('m.model') # 可以把模型储存下来以便下游任务使用

    my_matrix = np.loadtxt(open("Facebook.csv"), dtype=int, delimiter=",", skiprows=0)
    X, y = my_matrix[:, :-1], my_matrix[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
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


    G = nx.DiGraph()
    G.add_nodes_from(node)
    G.add_edges_from(edge_train)

    G_all = nx.DiGraph()
    G_all.add_nodes_from(node)
    G_all.add_edges_from(edge_exit)

    edge_not_exit = list(nx.non_edges(G_all))

    model = deepwalk(G, dimensions = 128, walk_length = 100, num_walks = 800000, min_count = 3)

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
        j += 1600

    print(roc_auc_score(y_true, y_scores))