import numpy as np 
import random
import pandas as pd
import time
from  tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns

def gengraph(n=10, m=8,q=1,r=10**6,  directed_graph = False, random_seed=None):
    """
    n: integer, кол-во верчин
    m: integer, кол-во ребер
    q: integer, нижняя граница для весов ребер графа
    r: integer, верхняя граница для весов ребер графа
    directed_graph:  bool, Если True генерируем оринетированный граф, иначе для каждой вершины (u, v)  и (v, u) 
                     одинаковый вес.
    random_seed:  None or integer. Случайность, если задано число, то граф фиксируется.
    
    1. Сначала генериуем всевозможные ребра (сразу же исключая петли)
    2. Затем исходя из значения directed_graph генерируем либо ориентированный либо обыкновенный граф
    
    return: V - множество вершин, E - список ребер, W - список весов (для ориентированного графа) 
       
    """
    V = set(range(n))
    Emax = []
    for v in V:
        for v1 in V:
            #сразу проверим, чтобы петлей не было
            if v!=v1:
                Emax.append((v,v1))
    if directed_graph:
        # Выберим m ребер
        random.seed(random_seed)
        E = random.sample(Emax,m)  
        #Сгенерируем псевдослучайные веса для ребер 
        random.seed(random_seed)
        W = random.choices(range(q, r), k=len(E))
    else:  
        #подбираем ребра для обыкновенного графа
#         E = [Emax[i] for i in range(len(Emax)-1) if Emax[i][-1::-1] not in Emax[i+1:]]
#         E.append(Emax[-1])
        random.seed(random_seed)
        E = random.sample(Emax,m)
        #Сгенерируем псевдослучайные веса для ребер 
        random.seed(random_seed)
        W = random.choices(range(q, r), k=len(E))
        #Cделаем граф ориентированным. Веса для вершин (v,u) и (u,v) сделаем ровным
        temp_E=[]
        temp_W = []
        for i in range(len(E)):
            if E[i][-1::-1] in E[i]:
                ind = E.index(E[i][-1::-1])
                W[i] =W[ind]
            else:
                temp_E.append(E[i][-1::-1])
                temp_W.append(W[i])
        E.extend(temp_E)
        W.extend(temp_W)
    return V, E, W
    
def timing(f):
    '''
    Measures time of function execution
    '''
    def wrap(*args, **kw):
        time1 = time.time()
        ret = f(*args, **kw)
        time2 = time.time()

        return (ret[0],ret[1], round(time2 - time1,2))
    return wrap


class Graph():
    def __init__(self, n=10, m=5,q=1,r=10**6, directed_graph = True,random_seed=1):
        self.V, self.E,self.W = gengraph(n=n,
                                         m=m,
                                         q=q,
                                         r=r,
                                         directed_graph = directed_graph,
                                         random_seed=random_seed)
        self.neighbors = self.get_vertex_neighbors()
    def get_vertex_neighbors(self):
        neighbors = dict(zip(self.V,[set() for i in range(len(self.V))]))
        
        start_v  = list(map(lambda x: x[0],self.E))
        end_v  = list(map(lambda x: x[1],self.E))        
        for i in range(len(start_v)):
            neighbors[start_v[i]].add(end_v[i])
        neighbors ={v:list(neighbors[v]) for v in neighbors}
        return neighbors

class MyNx():
    def __init__(self, graph):
        self.graph = graph
        self.E =graph.E
        self.V = graph.V
        self.W = graph.W
        self.neighbors = graph.neighbors
        self.E_W = dict(zip(self.E,self.W))
        pass
    @timing
    def dijkstra_predecessor_and_distance(self, source):
        """
            Алгоритм Дейкстри на основе метках
            
            source: integer - узел, для которого нужно найти кратчайщие пути от остальных узлов
            
            return: возвращает массив TETA[1..n], (TETA[i] – предпоследняя вершина в построенном
                    кратчайшем пути из вершины source в вершину i).
                    LABEL[1..n], (LABEL[i] – кратчайшее расстояние от вершины source до вершины i)
        """
        E = self.E
        V = self.V
        W = self.W
        neighbors = self.neighbors        
        E_W = self.E_W
        
        START_V = [source]
        TETA = [None for i in range(len(V))]#ЕСЛИ TETA[i] = j,означает, что узла i предшествует j (т.е. сущ. дуга (j, i))
        label = [np.inf]*len(V)# Метки узел. label[i] = value, озночает, что от узла source до i расстояние = value
        """CLOSE_V, метки для закрытии вершин (если для конкретного узла рассмотрели все его соседи, то отметим его 1)"""
        CLOSE_V = [0 for i in range(len(V))]
        iter_= 0 
        while len(START_V)>0:#пока все вершины не посещены продолжаем итерацию    
#             print(iter_)
            if iter_==0:
                label[START_V[0]] = 0 
            v_start_candidates = []#cобираем всех соседей рассмотриваемых узлов 
            for v_st in START_V:
                #Сортируем узлы по возрастанию весов ребер
                if iter_== 0:
                    neighbors[v_st] = sorted(neighbors[v_st],key=lambda x: E_W[(v_st, x)])
        #         Сортируем узлы по возрастанию меток.
                else:            
                    neighbors[v_st] = sorted(neighbors[v_st],key=lambda x: label[v_st])
        #         проходим по всем соседям текущего узла
                for neigh in neighbors[v_st]:
                    # обновим метку (расстояние) и предществующий узел,если найденное расстояние меньше чем значение метки
                    if label[v_st]+E_W[(v_st, neigh)]<label[neigh]:
                        label[neigh] = label[v_st]+E_W[(v_st, neigh)]
                        TETA[neigh] = v_st
                #Все соседей добавим, чтобы на след итерации пройти по ним 
                v_start_candidates.extend(neighbors[v_st])
                # Отметим текущий узел посещенным
                CLOSE_V[v_st]=1
            # Оставим только не посещенные узлы
            START_V = [v for v in v_start_candidates if CLOSE_V[v]==0] 
            del v_start_candidates
            iter_+=1
        return TETA, label


def experiment(N, M, q, r,m_label = 'm=n^2/10', source = 8):
    Time = []
    for i in tqdm_notebook(range(len(N))):
#         m = round(n**2/10)
        g = Graph(n=N[i],
                  m=M[i],
                  q=q[i],
                  r=r[i],
                  directed_graph = True,
                  random_seed=321121)
        mynx = MyNx(g)
        TETA, label,time_ = mynx.dijkstra_predecessor_and_distance(source=source)
        Time.append(time_)
    return pd.DataFrame({'n':N, 'm':M, 'Time, sec':Time,'r':r, 'label':[m_label]*len(Time)})