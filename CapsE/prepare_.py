import codecs
import numpy as np
from sklearn.model_selection import train_test_split


def get_data_for_net_ARXIV(embeddings_file, triples_file, SIZE=200):
    
    # сначала выбрали доки, у которых вообще есть эмбеддинги
    have_embeddings = set()
    with codecs.open(embeddings_file) as f:
        for line in f:
            id_, _ = line.split(' ', maxsplit=1)
            have_embeddings.add(id_[2:])
            
    # выбрали уникальные айдишники запросов и ответов, у которых есть эмбеддинги
    query_ids_unique = set()
    doc_ids_unique = set()

    with codecs.open(triples_file) as f:
        for line in f:
            id1, id2, id3 = map(lambda x: x.split('/pdf/')[-1], line.split())
            if id1 in have_embeddings and id2 in have_embeddings and id3 in have_embeddings:
                query_ids_unique.add(id1)
                doc_ids_unique.add(id2)
                doc_ids_unique.add(id3)
        
    query_ids_unique = list(query_ids_unique)
    doc_ids_unique = list(doc_ids_unique)
    
    # составили словари id-index и наоборот
    query_indexes = {}
    indexes_query = {}

    doc_indexes = {}
    indexes_doc = {}

    for i in range(len(query_ids_unique)):
        id_ = query_ids_unique[i]
        query_indexes[id_] = i
        indexes_query[i] = id_
    
    for i in range(len(doc_ids_unique)):
        id_ = doc_ids_unique[i]
        doc_indexes[id_] = i
        indexes_doc[i] = id_
        
    # получаем пары, которые будем пускать в сетку
    all_duplets = []

    with codecs.open(triples_file) as f:
        for line in f:
            id1, id2, id3 = map(lambda x: x.split('/pdf/')[-1], line.split())
            if id1 in have_embeddings and id2 in have_embeddings and id3 in have_embeddings:
                idx1 = query_indexes[id1]
                idx2 = doc_indexes[id2]
                idx3 = doc_indexes[id3]
                all_duplets.append([[idx1, idx2], [idx1, idx3]])
                
    train_duplets, test_duplets = train_test_split(all_duplets, test_size=0.3, random_state=42)
    train_duplets = np.array(train_duplets)
    test_duplets = np.array(test_duplets)
    
    train_val_duplets = np.array([np.array([[1], [-1]]) for i in range(len(train_duplets))])
    test_val_duplets = np.array([np.array([[1], [-1]]) for i in range(len(test_duplets))])
    
    # список эмбеддингов для запросов и доков
    lst_embeddings_query = np.zeros((len(query_ids_unique), SIZE))
    lst_embeddings_doc = np.zeros((len(doc_ids_unique), SIZE))

    with codecs.open(embeddings_file) as f:
        for line in f:
            id_, embedding = line.split(' ', maxsplit=1)
            embedding = list(map(np.float32, embedding.split()))
            id_ = id_[2:]
            if id_ in query_indexes:
                lst_embeddings_query[query_indexes[id_]] = embedding
            
            if id_ in doc_indexes:
                lst_embeddings_doc[doc_indexes[id_]] = embedding
                
    lst_embeddings_query = np.array(lst_embeddings_query, dtype=np.float32)
    lst_embeddings_doc = np.array(lst_embeddings_doc, dtype=np.float32)
    
    return lst_embeddings_query, lst_embeddings_doc, \
           train_duplets, test_duplets, \
           train_val_duplets, test_val_duplets


def get_duplets_and_val_duplets(behaviors, embeddings_file, have_embeddings, query_indexes, doc_indexes):
    
    duplets = []
    val_duplets = []
    with codecs.open(behaviors) as f:
        for line in f:
            _, _, _, hist, impressions = line.split('\t')
            queries = [i for i in hist.split() if i in have_embeddings]
            
            impressions = [(i.split('-')[0], int(i.split('-')[1])) for i in impressions[:-1].split()]
            impressions.sort(key=lambda x: x[1], reverse=True)
            
            docs_old = [i[0] for i in impressions]
            docs = [i for i in docs_old if i in have_embeddings]
            
            ranks = [i[1] for i in impressions]
            ranks = np.array([[ranks[i]] for i in range(len(ranks)) if docs_old[i] in have_embeddings])
            
            if len(docs) < 10 or len(queries) == 0:
                continue
                
            if np.sum(ranks) < 3 or np.sum(ranks) > 7:
                continue

            docs = docs[:10]
            ranks = ranks[:10]
            
            #переделываем пары айдишников в пары индексов
            queries = np.array([query_indexes[id_] for id_ in queries])
            docs = np.array([doc_indexes[id_] for id_ in docs])
            
            duplets.extend(np.transpose([np.repeat([i], 10), np.tile(docs, 1)]) for i in queries)
            val_duplets.extend(np.repeat(ranks, len(queries)).reshape((len(queries), 10, 1), order='F'))
            
    return np.array(duplets), np.array(val_duplets)


def get_data_for_net_MIND(embeddings_file_train, 
                          embeddings_file_test,
                          behaviors_train,
                          behaviors_test,
                          SIZE=200):
    
    have_embeddings = set()
    with codecs.open(embeddings_file_train) as f:
        for line in f:
            id_, _ = line.split(' ', maxsplit=1)
            have_embeddings.add(id_)
        
    with codecs.open(embeddings_file_test) as f:
        for line in f:
            id_, _ = line.split(' ', maxsplit=1)
            have_embeddings.add(id_)
            
    query_ids_unique = set()
    doc_ids_unique = set()
    
    with codecs.open(behaviors_train) as f:
        for line in f:
            _, _, _, hist, impressions = line.split('\t')
            queries = set(i for i in hist.split() if i in have_embeddings)
            
            docs_old = [i.split('-')[0] for i in impressions[:-1].split()]
            docs = set(i for i in docs_old if i in have_embeddings)
        
            query_ids_unique = query_ids_unique | queries
            doc_ids_unique = doc_ids_unique | docs
        
    with codecs.open(behaviors_test) as f:
        for line in f:
            _, _, _, hist, impressions = line.split('\t')
            queries = set(i for i in hist.split() if i in have_embeddings)
            
            docs_old = [i.split('-')[0] for i in impressions[:-1].split()]
            docs = set(i for i in docs_old if i in have_embeddings)
        
            query_ids_unique = query_ids_unique | queries
            doc_ids_unique = doc_ids_unique | docs

    query_ids_unique = list(query_ids_unique)
    doc_ids_unique = list(doc_ids_unique)
    
    query_indexes = {}
    indexes_query = {}

    doc_indexes = {}
    indexes_doc = {}

    for i in range(len(query_ids_unique)):
        id_ = query_ids_unique[i]
        query_indexes[id_] = i
        indexes_query[i] = id_
    
    for i in range(len(doc_ids_unique)):
        id_ = doc_ids_unique[i]
        doc_indexes[id_] = i
        indexes_doc[i] = id_
        
    train_duplets, train_val_duplets = get_duplets_and_val_duplets(behaviors_train, embeddings_file_train, have_embeddings,
                                                                  query_indexes, doc_indexes)
    test_duplets, test_val_duplets = get_duplets_and_val_duplets(behaviors_test, embeddings_file_test, have_embeddings,
                                                                query_indexes, doc_indexes)
    
    # список эмбеддингов для запросов и доков
    lst_embeddings_query = np.zeros((len(query_ids_unique), SIZE))
    lst_embeddings_doc = np.zeros((len(doc_ids_unique), SIZE))

    with codecs.open(embeddings_file_train) as f:
        for line in f:
            id_, embedding = line.split(' ', maxsplit=1)
            embedding = list(map(np.float32, embedding.split()))
            if id_ in query_indexes:
                lst_embeddings_query[query_indexes[id_]] = embedding
            
            if id_ in doc_indexes:
                lst_embeddings_doc[doc_indexes[id_]] = embedding

    with codecs.open(embeddings_file_test) as f:
        for line in f:
            id_, embedding = line.split(' ', maxsplit=1)
            embedding = list(map(np.float32, embedding.split()))
            if id_ in query_indexes:
                lst_embeddings_query[query_indexes[id_]] = embedding
            
            if id_ in doc_indexes:
                lst_embeddings_doc[doc_indexes[id_]] = embedding
                
    lst_embeddings_query = np.array(lst_embeddings_query, dtype=np.float32)
    lst_embeddings_doc = np.array(lst_embeddings_doc, dtype=np.float32)
    
    # вот тут нужно поколдовать насчет выборки не первых, а случайных
    indexes_train = np.arange(len(train_duplets))
    choice_train = np.random.choice(indexes_train, 20000)
    
    train_duplets = np.array([train_duplets[i] for i in choice_train])
    train_val_duplets = np.array([train_val_duplets[i] for i in choice_train])
    
    indexes_test = np.arange(len(test_duplets))
    choice_test = np.random.choice(indexes_test, 6000)

    test_duplets = np.array([test_duplets[i] for i in choice_test])
    test_val_duplets = np.array([test_val_duplets[i] for i in choice_test])
    
    return lst_embeddings_query, lst_embeddings_doc, \
           train_duplets, test_duplets, \
           train_val_duplets, test_val_duplets