
# coding: utf-8

# # L4 - Word embedding
# 
# **Векторное представление слов (word embedding)** - общее название для различных подходов к моделированию языка и обучению представлений в обработке естественного языка, направленных на сопоставление словам из некоторого словаря векторов из $\mathbb{R}^n$, где $n$ значительно меньше размера словаря. Показано, что векторные представления слов пособно значительно улучшить качество работы некоторых методов автоматической обработки естественного языка, например, синтаксический анализ или анализ тональности.
# 
# Обычно задача нахождения представленя сводится к обучению без учителя на большом корпусе текста. А процесс обучения основывается на том, что контекст слова тестно связан с ним самим. К примеру, контекстом может быть документ, в котором это слово находится, или соседние слова.

# In[1]:

import gensim.matutils
from gensim.corpora import WikiCorpus, MmCorpus, Dictionary
from gensim.models import TfidfModel
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn.decomposition
from collections import Counter, defaultdict
from random import shuffle
import collections
import random
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output
from mpl_toolkits.mplot3d import Axes3D

random.seed(0xFEEDF00D)
np.random.seed(0xDEADBEEF)
tf.set_random_seed(0xCAFEBABE)
get_ipython().magic('matplotlib inline')


# ## Данные
# 
# **Задание**
# 1. Удобным для вас способом выкачайте себе обучающий корпус текста.
# 2. Подговьте статьи для процесса обучения, в частности нужно убрать всю разметку. Поработать с редко встречающимися словами, тут бывают разные техники.
# 
# В качестве корпуса будем использовать дамп [Simple English Wikipedia](https://dumps.wikimedia.org/simplewiki/20170201/simplewiki-20170201-pages-articles.xml.bz2) от 1 февраля 2017 года, предобработка статей осуществляется посредством библиотеки gensim. При этом после парсинга отбрасываются все слова, встречающиеся чаще, чем в 15% документов и реже, чем 10 раз.

# In[2]:

outp='simplewiki'
wiki = WikiCorpus(outp+'-20170201-pages-articles.xml.bz2', lemmatize=False)
wiki.dictionary.filter_extremes(10,0.15,keep_n=100000)
tfidf = TfidfModel(wiki, id2word=wiki.dictionary, normalize=True)
vm=set(map(lambda x:x.encode(),wiki.dictionary.values()))
vocabulary_size = len(wiki.dictionary)


# ## LSA (Latent semantic analysis)
# 
# **Задание**
# 1. Вычислите $X$, понравившимся способом (возможно, стоит использовать разреженную матрицу).
# 2. Обучите представлении слов $k = 128$ на своем корпусе.
# 3. Реализуйте поиск k-ближайших соседей для Евклидовой меры в 128-мерном пространстве .
# 4. Найдите 10 ближайших представлений к 30 словам, которые выбрали сами. Попытайтесь сделать так, чтобы примеры были интересными.
# 5. Проделайте то же самое для косинусной меры. Какой результат вам показался более интересным?
# 6. Предложите свою меру длины, проверьте, как она работает.

# In[3]:

svd=sklearn.decomposition.TruncatedSVD(n_components=128)
emb_svd=svd.fit_transform(gensim.matutils.corpus2csc(tfidf[wiki],num_docs=wiki.length,num_terms=vocabulary_size))
del tfidf


# In[7]:

def get_nn(word,matrix,id2word,word2id,metric="cosine",k=10):
    ident=word2id(word)
    v=matrix[ident].reshape(1,matrix.shape[1])
    dists=cdist(matrix,v,metric).flatten()
    return [id2word(i) for i in np.array(dists).argsort()[:k+1] if i!=ident][:k]

def get_sub_add_nn(word,sub,add,matrix,id2word,word2id,metric="cosine",k=10):
    rest_ind=[word2id(x) for x in [word,sub,add]]
    v=(matrix[word2id(word)]-matrix[word2id(sub)]+matrix[word2id(add)]).reshape(1,matrix.shape[1])
    dists=cdist(matrix,v,metric).flatten()
    return [id2word(i) for i in dists.argsort()[:k+3] if i not in rest_ind][:k]

def print_neighbors(word_list,matrix,id2word,word2id,k=10):
    metrics=["Euclidean","Cosine","Manhtattan"]
    for word in word_list:
        neighbors= [get_nn(word,matrix,id2word,word2id,"euclidean",k),
                    get_nn(word,matrix,id2word,word2id,"cosine",k),
                    get_nn(word,matrix,id2word,word2id,"cityblock",k)]
        neighbors=list(zip(metrics,[", ".join(l) for l in neighbors]))
        print("Nearest to "+word+":")
        print("\n".join(": ".join(neighbor) for neighbor in neighbors),end='\n\n')


# Наряду с евклидовой и косинусной рассмотрим манхэттенскую метрику:
# $$\sum_{i=1}^n |p_i-q_i|$$
# 
# Сравним соседей 30 слов сразу во всех трёх метриках:

# In[8]:

word_list=['king','queen','russia','cat','skillful','rock','mean','variance','default','set','actor',
           'era','performance','thousand','general','facebook','apple','apple','them','couple','river',
           'everest','player','lasso','cube','article','exoplanet','handgun','bus','word']


# In[9]:

print_neighbors(word_list,emb_svd,lambda x: wiki.dictionary[x],lambda x: wiki.dictionary.doc2bow([x])[0][0])


# ## word2vec
# 
# Теперь рассмотрим, пожалуй, одну из самых популярных моделей представления слов. В основном это связано с тем, что полученные вектора обладают интересными свойствами. В частности, похожие по смыслу слова являются близкими векторами, более того линейная операция вычитания имеют смысл в полученном векторном прострастве! Контекстом для слова на этот раз является некоторое скользящее окно размера $2c+1$, где интересующее нас слово занимает среднюю позицию.
# 
# ### Предсказание слова по контексту (Continuous bag-of-words)
# Идея, на которой основывается модель, очень проста. Пусть у нас есть корпус текста
# $$w_1, w_2, \dots, w_T.$$
# Рассмотрим некоторое слово $w_t$ и его контекст радиуса $c$, а именно слова 
# $$w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots w_{t+c}.$$
# Интуиция нам подсказывает, что контекст очень хорошо характеризует слово. Поэтому мы попытаемся обучить такую модель, которая по контесту восстаналивает центральное слово
# $$\mathcal{L} = - \frac{1}{T}\sum_{t} \mathbb{P}[w_t|w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots w_{t+c}].$$
# 
# Пусть для каждого слова $w_t$ есть некоторое представление $v_t$, а когда слово $w_t$ входит в чей-то контекст, то мы будем кодировать его с помощью $v'_{t}$. Тогда весь контекст слова $w_t$ можно описать суммой
# $$s'_t = \sum_{c \leq i \leq c, i \neq 0} v'_{t+i}$$
# 
# Таким образом искомую вероятность мы будем приближать следующей моделью
# $$\mathbb{P}[w_t|w_{t-c}, \dots, w_{t-1}, w_{t-1}, \dots w_{t+c}] = \frac{ \exp {s'}_t^T \cdot v_t}{\sum_j \exp {s'}_t^T \cdot v_j}.$$
# 
# Заметим, что подобную модель несложно представить в виде простейшей нейронной сети:
# * На вход подается разреженный вектор, где единичка стоят на позициях, соответствующих словам контекста.
# * Далее идет полносвязанный слой, который состоит из векторов $v'$.
# * Далее без применения какой-либо нелинейности идет матрица, состоящая из $v$.
# * А затем слой softmax.
# 
# ### Предсказание контекста по слову (Skip-gram)
# Другая модель, которая чаще применяется на практике, пытается по слову предсказать его контекст. Конструкция примерно та же самая, нужно оптимизировать
# $$\mathcal{L} = \frac{1}{T} \sum_t \sum_{-c \leq i \leq c, i \neq 0}\log \mathbb{P}[w_{t+i}|w_t].$$
# 
# При этом вероятность аппроксимируется следующей моделью
# $$\mathbb{P}[w_{t+i}|w_t] = \frac{ \exp {v'}_{t+i}^T \cdot v_t}{\sum_j \exp {v'}_{j}^T \cdot v_t}.$$
# 
# **Задание**
# 1. Как можно представить модель skip-gram в виде нейронной сети?
# 2. Оцените сложность обучения skip-gram модели относительно параметров 
#   * $T$ - размер корпуса
#   * $W$ - размер словаря 
#   * $c$ - радиус окна
#   * $d$ - размерность представлений

# 1. Модель практически аналогична варианту CBOW, основные отличия истекают из того, что мы теперь предсказываем контекст:
#     * Входной вектор - разреженный, единица стоит на позиции, ему соответствующей.
#     * Первым полносвязным слоем мы получаем $h:=v_{w_i}^T$ - вектор, полученный умножением матрицы $W$ на входной. Из свойств входного вектора получаем, что данный вектор является искомым представлением слова. 
#     * Затем, рассматривая некоторое число контекстов, мы вначале умножаем $h$ на ещё одну матрицу $\widetilde{W}$ ("матрицу контекстов"), а затем используем softmax для определения вероятности нахождения тех или иных слов в контексте входного. 
#     * Функция ошибки в данном случае - среднее арифметическое суммы логарифмов правдоподобия нахождения слов в контексте входного по всем словам из входной выборки.
# 2. Если считать, что значения softmax известны после прохода вперёд, получаем следующую сложность:
#     * Перемножение матриц: $1\times W \cdot W\times D$ для первого слоя, $1\times D \cdot D\times W$ для второго - сложность $O(WD)$.
#     * Softmax: подсчёт $O(W)$, так как произведение уже выполнено и остаётся только разделить его на сумму экспонент всех остальных элементов вектора. Однако нам необходимо просуммировать их логарифмы для $O(C)$ слов из контекста, поэтому на данном этапе получаем $O(WC)$.
#     * При обучении нам важны производные функции ошибки по двум матрицам представлений. Используя найденную ранее производную softmax, а также уравнения для обратного распространения ошибки, получаем, что в формулу будет входить произведение вектора $v_t$ и softmax всех $C$ контекстов, что даст нам операцию сложности $O(WCD)$, которая и будет доминирующей при вычислении.
#     * Так как процесс необходимо повторить для каждого элемента в корпусе (или каждого батча, число которых также по логике должно быть пропорционально их размеру), асимптотическая сложность алгоритма дополнительно умножается на $T$. Получаем $O(TWCD)$.

# ### Аппроксимация softmax
# 
# На самом деле Миколов, создавая свою модель, руководствовался тем, чтобы сократить сложность обучения. Однако вычисление знаменателя в softmax является дорогой операцией, так как необхоидмо проходить по всему словарю. Поэтому на практике обучаются несколько иные модели, которые пытаются аппроксимировать поведение softmax.
# 
# #### Negative Sampling
# 
# Данная функция потерь является упрощением **Noise contrastive estimation**, которая в свою очередь является аппроксимацией softmax. Суть в том, что можно попробовать оптимизировать функцию, сильно похожу на логистическую регрессию.
# 
# $$\mathcal{L} = - \frac{1}{T}\sum_{w_t}
#     \sum_{-c \leq i \leq c, i \neq 0} \Big(
#         \mathbb{P}[y = 1 | w_{t+i}, w_t] -
#         k \cdot \mathbb{E}_{\tilde{w}_{tj} \sim \mathbb{P}[w_j|w_t]} \mathbb{P}[y = 0 | \tilde{w}_{tj}, w_t],
#     \Big)
# $$
# где $y=1$, если одно слово является контекстом другого и 0 в обратном случае. А $\tilde{w}_{tj}$ - это специально семплированный слова, который не принадлежат контексту.
# 
# 
# Дальше вся игра заключается в том, как мы распишем соответствующее распределение $\mathbb{P}[y | w_{t+i}, w_t]$. В [noise contrastive estimation](https://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann10AISTATS.pdf) есть свои навороты. Миколов же предлагает оптимизировать
# 
# $$\mathcal{L} = - \frac{1}{T}\sum_{w_t} \Big(
#     \sum_{-c \leq i \leq c, i \neq 0} \big(
#         \sigma({v'}_{t+i}^T v_t) +
#         k \cdot \mathbb{E}_{w_j \sim \mathbb{Q}(w_j)} \sigma (-{v'}_{j}^T v_t)
#     \big)
# \Big),
# $$
# 
# где для вычисления математического ожидания мы просто делаем $k$ семплов из какого-то распределения $\mathbb{Q}$ на словах, то есть $\mathbb{P}(w_j|w_t) = \mathbb{Q}(w_j)$.
# 
# 
# **Задание**
# 1. Выберите понравивщуюся вам функцию потерь. Оцените сложность обучения для нее, сравните с простым softmax.
# 2. На основе оценки сложности определиться с количеством эпох, размера вектора $d$ = 256 будет достаточно. Определитесь с размером окна. Будьте готовы, что на большом корпусе обучение может длиться около суток.
# 2. Обучите skip-gram модель.
# 3. Попробуйте снова найти ближайшие представления для тех 30 слов. Улучшился ли результат визуально? Попробуйте разные меры расстояния (евклидова, косинусная).
# 4. Найдите ближайшие вектора для выражения v(king) - v(man) + v(women). Если модель обучена хорошо, то среди ближайших векторов должно быть представление v(queen). 
# 

# Оценим сложность при обучении Noise Contrastive Estimation:
# 
#   * Зависимость от $T,C,D$ практически аналогичная, а так как умножение матриц будет являться неотъемлемой частью любой модели, далее исключим включающее в себя $W$ выражение из оценки сложности.
#   * Важнее отсутствие softmax, заменённое отбором $k$ примеров не из контекста рассматриваемого слова. Так как $k<<W$ на практике, то данный этап будет работать за $O(k)$, а значит, значительно быстрее, и итоговая сложность составит $O(TCDk)$.

# In[10]:

def build_dataset():
    count=[]
    counter=collections.Counter()
    dictionary = {}
    data = []
    for line in wiki.get_texts():
        l=[i.decode() for i in line if i in vm]
        counter.update(l)
        data.extend(l)
    count.extend(counter.most_common(vocabulary_size))
    dictionary={count[i][0]:i for i in range(len(count))}
    data=[dictionary[word] for word in data]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset()

data_index = 0


# In[11]:

batch_size = 256
embedding_size = 256
skip_window = 3
num_skips = 4

valid_size = 16
valid_window = 100  
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 

graph = tf.Graph()

with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / np.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,
                                         inputs=embed,num_sampled=num_sampled,num_classes=vocabulary_size))

    optimizer = tf.train.AdagradOptimizer(1).minimize(loss)

    init = tf.global_variables_initializer()


# In[12]:

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    for i in range(batch_size // num_skips):
        available_targets=list(range(len(buffer)))
        available_targets.remove(skip_window)
        targets=random.sample(available_targets,num_skips)
        
        for j,target in enumerate(targets):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# In[13]:

num_steps = 500000

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    average_loss = 0
    
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 100000 == 0:
            if step > 0:
                average_loss /= 100000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
    final_embeddings = embeddings.eval()


# In[14]:

print_neighbors(word_list,final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[15]:

get_sub_add_nn('king','man','woman',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# "Queen" оказалось аж вторым в списке соседей, причём остальные слова также семантически близки к тем, над которыми совершались операции

# На основе арифметических операций над представлениями предложите алгоритмы, которые
#   * Для страны определяют столицу
#   * Для компании определяют CEO
#   * Для прилагательного определяют превосходную форму
#   * Для слова определяют множественное число
#   * Для слова находит его антоним
#   
# Так как основная операция над данными векторами - операция нахождения аналогии, то можно предложить соответственно следующие алгоритмы, а большую их часть даже и проверить:
#    * Moscow-Russia+[country] (вычитая соответствующий стране вектор, надеемся получить вектор, который при прибавлении к названию другой страны окажется близко к названию её столицы). Все остальные алгоритмы были разработаны из тех же соображений вычитания некоторой компоненты, чтобы найти после добавления другой компоненты семантически состоящий в похожем отношении с ней аналог исходного слова.
#    * Zuckerberg-Facebook+[company]. К сожалению, в Simple English версии Википедии, по всей видимости, слишком мало упоминаний CEO каких-либо известных компаний, поэтому данный алгоритм проверить не удалось (однако логика его работы аналогична остальным)
#    * largest-large+[adjective]
#    * cats-cat+[noun]
#    * small-big+[word]
#    
# Заметим, что формирующие данные смысловые отношения пары слов не единственны, поэтому вполне возможно, что на другой паре алгоритм будет давать более точные или устойчивые при испытании на большом числе экземпляров результаты.

# In[16]:

get_sub_add_nn('moscow','russia','germany',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[17]:

get_sub_add_nn('athens','greece','japan',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[18]:

get_sub_add_nn('largest','large','big',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[19]:

get_sub_add_nn('small','big','fast',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[20]:

get_sub_add_nn('cats','cat','tree',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[21]:

get_sub_add_nn('dorsey','twitter','google',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# Как можно видеть, все алгоритмы кроме поиска CEO справляются с поставленной задачей хотя бы на одном примере. Также можно развить тему, создав, например, алгоритм получения произвольной формы слова (полезно с неправильными глаголами)

# In[22]:

get_sub_add_nn('brought','bring','come',final_embeddings,lambda x: reverse_dictionary[x],lambda x: dictionary[x])


# In[23]:

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=(50,50),dpi=200) 
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, alpha=0)
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', ha='center',
                     va='center')
    if filename is not None:
        figure.savefig(filename)
        plt.close(figure)


# In[24]:

tsne = TSNE(n_components=2,perplexity=10,n_iter=1000,init='pca',random_state=0x5EED1E55)
plot_only = 1000

low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only])
labels = [reverse_dictionary[i] for i in range(plot_only)]

plot_with_labels(low_dim_embs, labels,filename='tsne_w2v.png')

Image(filename='tsne_w2v.png') 


# ## GloVe
# 
# Данная модель пытается объединить в себе две идеи:
# 1. Факторизация матрицы, как в LSA
# 2. А вот контекстом слова является скользящее окно, как в v2w.
# 
# Пусть есть некоторая матрица $X$, где элемент $x_{ij}$ - это количество, сколько раз слово $j$ встрелось в контексте слова $i$. Эта матрица заполняется на основе нашего корпуса. Тогда вероятность, что слово $j$ появляется в контексте слова $i$ равна
# 
# $$P_{ij} = \frac{X_{ij}}{\sum_k X_{ik}}$$
# 
# При этом мы хотели бы найти такую функцию $F$ и представления для слов, что
# $$F((w_i - w_j)^T w_k) = \frac{F(w_i^T w_k)}{F(w_j^T w_k)} = \frac{P_{ik}}{P_{jk}}$$
# мотивировку именно такой модели лучше посомтреть в оригинальной [статье](http://nlp.stanford.edu/pubs/glove.pdf).
# Возможное решение выглядит так
# 
# $$\exp(w_i^T w_j) = P_{ij}$$
# 
# Если немного поколдовать, то получится, что наша модели выглядит
# $$w_i^T w_j = \log X_{ij} - \log\big(\sum_k X_{ik}\big)$$
# 
# Не сложно заметить, что второй член не зависит от $j$, однако итоговую модель все же предлагается для симметричности записать как
# 
# $$w_i^T w_j + b_i + b_j = \log X_{ij}$$
# 
# В качестве функции потерь можно использовать взвешенное квадратичное отклонение
# $$\mathcal{L} = \sum_{i,j = 1}^{|W|} f(X_{ij}) \Big(w_i^T w_j + b_i + b_j - \log(X_{ij}) \Big)^2,$$
# где функция $f$ определяет вес каждой пары слов. Разумно хранить матрицу $X_{ij}$ в разреженном виде. При этом примечательно то, что во время обучения авторы используют **только не нулевые** значения $X_{ij}$. Обучение ведется с помощью градиентного спуска, когда ненулевые значения просто семплируются из $X_{ij}$.
# 
# **Задание**
# 1. Оцените сложность модели. На основе оценки сложности поймите, какое $d$ и $c$ для вас оптимально.
# 2. Попробуйте обучить модель GloVe.
# 3. Убедитесь, что опять вектор v(king) - v(man) + v(women) ближе всего к v(queen).
# 4. Самостоятельно разберитесь, что такое [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).
# 5. Используя t-SNE сожмите представления до 3-мерных векторов. Убедитесь, что следующие группы векторов примерно сонаправлены
#   * [мужчина, женщина] (к примеру, [man, woman], [Mr., Ms], [king, queen] и.т.д)
#   * [CEO, компания]
#   * [прилагательное, его сравнительная форма]
# 5. Предложите метрику сравнения моделей (ее можно подсмотреть ;)).
# 6. Попробуйте семплировать также и нулевые значения матрицы $X_{ij}$. Ухудшилась ли модель?
# 

# Сложность формирования матрицы $X$: $O(TC)$, так как мы проходим по всем $T$ словам из корпуса, для каждого рассматриваем $O(C)$ соседей и обновляем соответствующие таким парам ячейки матрицы.
# Сложность обучения:
# 
#    * вычисление $\mathcal{L}$: введём параметр $|X|$, обозначающий число ненулевых элементов в матрице $X$. Тогда нахождение значения функции потерь имеет сложность $|X|\cdot O(WD)+O(D)$ ($|X|$ пар слов, для каждой необходимо за $O(WD)$ найти оба представления слов, затем за $O(D)$ перемножить их, остальные этапы выполняются за константное время. Таким образом, на этом этапе получаем $O(|X|WD)$.
#     
#    * обратное распространение ошибки: необходимо для каждого из $W$ слов найти градиент функции ошибки по обоим представлениям. Если считать, что результаты всех соответствующих скалярных произведений уже найдены при вычислении $\mathcal{L}$, а на каждой итерации цикла для отдельного слова потребуется дополнительно умножать на представление другого слова пары за $D$ шагов, то получаем опять же $O(W|X|D)$.
#     
# Таким образом, итоговая сложность составляет $O(TC+|X|WD)$. Остаётся вопрос: можно ли как-то оценить $|X|$? Очевидно, оценка в худшем случае - $W^2$, однако авторы статьи дополнительно выводят (с некоторыми допущениями) с помощью махинаций вида обобщённых гармонических чисел и дзета-функции Римана оценку в среднем вида $O(T)$, если параметр модели $\alpha<1$ и $O(T^{\frac{1}{\alpha}})$ иначе.
# 
# Семплирование же нулевых значений матрицы не сыграет никакой роли, за исключением того, что в функции потерь станет больше нулевых слагаемых ($f(0)=0$), то есть мы будем сдвигаться на меньшее значение, чем могли бы, по каждому из рассматриваемых в батче слов (представления слов из пары, для которой $X_{ij}=0$, не будут сдвигаться никуда в соответствующих друг другу слагаемых из-за нулевого значения этих слагаемых. Получаем, что это в каком-то смысле ухудшит модель, так как часть вычислений во время каждого цикла алгоритма будет выполняться впустую (скалярное произведение векторов слов, которые не встречаются друг рядом с другом).

# In[41]:

class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=0.75, cooccurrence_cap=100, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size
        self.left_context = self.right_context = context_size
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__cooccurrence_matrix = None
        self.__embeddings = None

    def fit_to_corpus(self):
        self.__fit_to_corpus(self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__build_graph()

    def __fit_to_corpus(self, vocab_size,min_occurrences,  left_size, right_size):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for line in wiki.get_texts():
            l=[i.decode() for i in line if i in vm]
            word_counts.update(l)
            for l_context, word, r_context in _context_windows(l, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        self.__cooccurrence_matrix = {
            (dictionary[words[0]], dictionary[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in dictionary and words[1] in dictionary}


    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32)
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32)

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size])

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0))
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0))

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0))
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0))

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            weighting_factor = tf.minimum(1.0,tf.pow(tf.div(self.__cooccurrence_count, count_max),
                                                     scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([embedding_product,focal_bias,context_bias,
                                                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.__total_loss)

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings)

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        batches = self.__prepare_batches()
        total_steps = 0
        average_loss = 0
        with tf.Session(graph=self.__graph) as session:
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    _, loss_val = session.run([self.__optimizer,self.__total_loss], feed_dict=feed_dict)
                    average_loss += loss_val
                    if total_steps % 500000 == 0:
                        if total_steps > 0:
                            average_loss /= 500000
                        print("Average loss at step ", total_steps, ": ", average_loss)
                        average_loss = 0
                    total_steps += 1
            self.__embeddings = self.__combined_embeddings.eval()

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[dictionary[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    @property
    def vocab_size(self):
        return len(reverse_dictionary)

    @property
    def words(self):
        if reverse_dictionary is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return reverse_dictionary

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if dictionary is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return dictionary[word]


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


# In[44]:

model=GloVeModel(embedding_size=192,context_size=6,min_occurrences=4,batch_size=256)


# In[46]:

model.fit_to_corpus()


# In[47]:

model.train(10)


# In[48]:

emb_glove=model.embeddings


# In[49]:

get_nn('cat',emb_glove,lambda x: model.words[x],model.id_for_word,"cosine")


# In[50]:

get_nn('april',emb_glove,lambda x: model.words[x],model.id_for_word,"euclidean")


# t-SNE - алгоритм понижения размерности данных, который достаточно неплохо сохраняет схожесть (в терминах расстояния Кульбака-Лейблера между распределениями) близких векторов в многомерном пространстве при их отображении в пространство меньшей размерности. Сделаем две визуализации (в двумерном и в трёхмерном пространствах, картинки лучше открывать в полном окне):

# In[59]:

low_dim_embs_glove = tsne.fit_transform(emb_glove[:10000])
plot_only = 200
labels_glove = [model.words[i] for i in range(plot_only)]

plot_with_labels(low_dim_embs_glove, labels_glove,filename='tsne_glove.png')

Image(filename='tsne_glove.png') 


# In[60]:

tsne3d=TSNE(n_components=3,perplexity=10,n_iter=5000,init='pca',random_state=0x5EED1E55)
embs3d=tsne3d.fit_transform(emb_glove[:10000])
fig = plt.figure(figsize=(20,20),dpi=200) 
ax = fig.add_subplot(111, projection='3d')
for i, label in enumerate(labels_glove):
    x, y,z = embs3d[i, :]
    ax.scatter(x, y,z, alpha=0)
    ax.text(x,y,z,label, ha='center',va='center')


# In[90]:

from scipy.spatial.distance import cosine
print(cosine(embs3d[dictionary['king']],embs3d[dictionary['queen']]))
print(cosine(embs3d[dictionary['apple']],embs3d[dictionary['jobs']]))
print(cosine(embs3d[dictionary['good']],embs3d[dictionary['better']]))


# Как видим, косинусное расстояние довольно небольшое (за исключением CEO, причины такой ситуации оговорены выше), поэтому можно сделать вывод, что векторы близких по смыслу слов действительно стали сонаправлены.

# 8 Сравните все три модели.
# 
# Воспользуемся датасетом [Google analogy test set](http://download.tensorflow.org/data/questions-words.txt) и измерим на нём точность полученных представлений следующим образом: если хотя бы один из 10 предложенных моделью вариантов совпадает с истинным, засчитывать 1, иначе - 0. Итоговой метрикой будет относительная точность.
# 
# 9 А теперь дайте остыть вашей ЖПУ и идите спать.

# In[61]:

questions=[]
vocab=set(wiki.dictionary.values())
with open('questions-words.txt') as f:
    for line in f:
        if not (line.startswith(':') or any(i not in vocab for i in line.split())):
            questions.append(list(map(lambda x: x.lower(),line.split())))


# In[69]:

def eval_svd():
    for question in questions:
        if question[2] in get_sub_add_nn(question[0],question[1],question[3],
                                            emb_svd,lambda x: wiki.dictionary[x],
                                            lambda x: wiki.dictionary.doc2bow([x])[0][0],k=10):
            arr[0]+=1

def eval_w2v():
    for question in questions:
        if question[2] in get_sub_add_nn(question[0],question[1],question[3],
                                            final_embeddings,lambda x: reverse_dictionary[x],
                                            lambda x: dictionary[x],k=10):
            arr[1]+=1

def eval_glove():
    for question in questions:
        if question[2] in get_sub_add_nn(question[0],question[1],question[3],
                                            emb_glove,lambda x: reverse_dictionary[x],
                                            lambda x: dictionary[x],k=10):
            arr[2]+=1


# In[70]:

from multiprocessing import Pool,Array
arr=Array('i',3,lock=False)
p=Pool(processes=3)
res_svd=p.apply_async(eval_svd)
res_w2v=p.apply_async(eval_w2v)
res_glove=p.apply_async(eval_glove)
p.close()
p.join()


# In[93]:

r=np.array(arr[:])/len(questions)


# In[94]:

r


# Выше приведены результаты LSA, Word2Vec и GloVe соответственно. Выходит, что Word2Vec не зря считается лидирующим алгоритмом, особенно если учесть скорость обучения значительно более высокую, чем у GloVe.
