from Bio import SeqIO

def w2v_train(seqs):
    from gensim.models import Word2Vec
    model = Word2Vec([list(seq) for seq in seqs], vector_size=1, min_count=1, window=1, workers=-1, epochs=128)
    model.save("word2vec.model")

def fasttest_train(seqs):
    from gensim.models import FastText
    model = FastText([list(seq) for seq in seqs], vector_size=1, min_count=1, window=1, workers=-1, epochs=128)
    model.save("fasttest.model")

def w2v_infer(seq,model_path):
    from gensim.models import Word2Vec
    model_w2c = Word2Vec.load(model_path)
    res = []
    seq_list = list(seq)
    for x in seq_list:
        tmp_w2v = model_w2c.wv[x]
        res.append(list(tmp_w2v))
    return np.array(res).mean().tolist()

def fasttext2vec_infer(seq,model_path):
    from gensim.models import FastText
    model_fast = FastText.load(model_path)
    res = []
    seq_list = list(seq)
    for x in seq_list:
        tmp_w2v = model_fast.wv[x]
        res.append(list(tmp_w2v))
    return np.array(res).mean().tolist()

def read_fa(path):
    res_p = {}
    res_n = {}
    rs = SeqIO.parse(path,format="fasta")
    for x in list(rs):
        id = str(x.id)
        if id[-1]=="1":
            seq = str(x.seq).upper()
            res_p[id]=seq
        else:
            seq = str(x.seq).upper()
            res_n[id]=seq
    return {**res_p,**res_n}

train_path = "../data/solubility_train_new.fa"
valid_path = "../data/solubility_dev_new.fa"
test_path =  "../data/solubility_test_new.fa"

train_seqs = list(read_fa(train_path).values())
valid_seqs = list(read_fa(valid_path).values())
test_seqs = list(read_fa(test_path).values())

seqs = train_seqs+valid_seqs+test_seqs

w2v_train(seqs)
fasttest_train(seqs)
