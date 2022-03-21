import joblib
from tqdm import tqdm
import scipy.sparse as sp
from collections import Counter
import numpy as np
from VISIBILITY import VISIBILITY_GRAPH 

# 数据集
dataset = "R8"
zaoyin=20
# 参数
window_size = 7
embedding_dim = 300
max_text_len = 800

# node_state=[i for i in range(25)]
# normalize
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized


def pad_seq(seq, pad_len):
    if len(seq) > pad_len: return seq[:pad_len]
    return seq + [0] * (pad_len - len(seq))


if __name__ == '__main__':
    # load data
    word2index = joblib.load(f"temp/{dataset}.word2index.pkl")
    with open(f"temp/{dataset}.texts.remove.txt", "r") as f:
        texts = f.read().strip().split("\n")

    # bulid graph
    inputs = []
    graphs = []
    input_sentence = []
    input_sentence_state = []
    inputs_sentence_state = []
    count_little_graph = []
    inputs_state = []
    graphs_state = []
    count = 1
    node_num = []
    edge_num = []
    for text in tqdm(texts):
        count = count + 1
        text2words = [word2index[w] for w in text.split()]
        nodes = list(set(text2words))
        print(nodes)
        print(len(nodes))
        node2index = {e: i for i, e in enumerate(nodes)}
        node_num.append(len(nodes))

        words = [[node2index[word2index[w]],len(w)] for w in text.split()]
        words = words[:max_text_len]  # 限制最大长度

        words_true=[node2index[word2index[w]] for w in text.split()]

        edge, all_graph_littel=VISIBILITY_GRAPH(words, 7)



        edge_state, state_nodes, count_little_graph, state_true=statenetwork(all_graph_littel, count_little_graph)
        edge_count = Counter(edge).items()
        print(edge_count)
        print("*****",len(edge_count))
        edge_num.append(len(edge_count))
        edge_state_count=Counter(edge_state).items()
        row = [x for (x, y), c in edge_count]
        col = [y for (x, y), c in edge_count]
        weight = [c for (x, y), c in edge_count]
        adj = sp.csr_matrix((weight, (row, col)), shape=(len(nodes), len(nodes)))
        adj_normalized = normalize_adj(adj)
        weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count]
        inputs.append(nodes)
        input_sentence.append(words_true)
        graphs.append([row, col, weight_normalized])



    len_inputs = [len(e) for e in inputs]
    len_inputs_sentence=[len(e) for e in input_sentence]

    len_graphs = [len(x) for x, y, c in graphs]
    pad_len_inputs = max(len_inputs)
    pad_len_inputs_sentence=max(len_inputs_sentence)
    pad_len_graphs = max(len_graphs)
    inputs_pad = [pad_seq(e, pad_len_inputs) for e in tqdm(inputs)]
    input_sentence_pad=[pad_seq(e,pad_len_inputs_sentence) for e in tqdm(input_sentence)]
    graphs_pad = [[pad_seq(ee, pad_len_graphs) for ee in e] for e in tqdm(graphs)]

    input_sentence_pad=np.array(input_sentence_pad)
    inputs_pad = np.array(inputs_pad)
    weights_pad = np.array([c for x, y, c in graphs_pad])
    graphs_pad = np.array([[x, y] for x, y, c in graphs_pad])
    node_num=np.array(node_num)
    edge_num=np.array(edge_num)





    all_vectors = np.load(f"source/glove.6B.{embedding_dim}d.npy")
    all_words = joblib.load(f"source/glove.6B.words.pkl")
    all_word2index = {w: i for i, w in enumerate(all_words)}
    index2word = {i: w for w, i in word2index.items()}
    word_set = [index2word[i] for i in range(len(index2word))]
    oov = np.random.normal(-0.1, 0.1, embedding_dim)
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else oov for w in word_set]
    word2vec.append(np.zeros(embedding_dim))
    state2vec = [oov for i,sta in enumerate(count_little_graph)]
    state2vec.append(np.zeros(embedding_dim))
    state2vec.append(np.zeros(embedding_dim))



    joblib.dump(len_inputs, f"temp/{dataset}.len.inputs.pkl")
    joblib.dump(len_graphs, f"temp/{dataset}.len.graphs.pkl")
    joblib.dump(len_inputs_sentence, f"temp/{dataset}.len.inputs_sentence.pkl")

    joblib.dump(len_graphs, f"temp/{dataset}.len.graphs.pkl")
    np.save(f"temp/{dataset}.inputs.npy", inputs_pad)
    np.save(f"temp/{dataset}.inputs_sentence.npy", input_sentence_pad)

    np.save(f"temp/{dataset}.graphs.npy", graphs_pad)
    np.save(f"temp/{dataset}.weights.npy", weights_pad)
    np.save(f"temp/{dataset}.word2vec.npy", word2vec)
    np.save(f"temp/{dataset}.nodes_num.npy", node_num)
    np.save(f"temp/{dataset}.edge_num.npy", edge_num)




