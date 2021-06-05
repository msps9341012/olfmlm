import faiss
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.manifold import MDS


class Neighbor_finder:
    def __init__(self, view):
        self.view = view
        self.index = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(self.view)
        self.index.add(view)
        
    def get_neighbor(self,text_id, n, text):
        
        q = np.expand_dims(self.view[text_id],axis=0)
        D, I = self.index.search(q, n)
        self.print_text(I,text, skip=True)
        return D, I
    
    def get_neighor_by_vector(self, q, n):
        q = np.expand_dims(q, axis=0)
        D, I = self.index.search(q, n)
        #self.print_text(I, skip)
        return D, I
        
    def print_text(self, I, text, skip=False):
        counter=1
        t=I[0].tolist()
        if skip:
            t.pop(0)
        
        for i in t:
            print("Neighbor "+str(counter)+" :"+ text[i])
            counter=counter+1
            
class Finder_by_Method:
    def __init__(self,method, text):
        self.method=method
        self.raw_text=text
        self.view_1=np.load('embeds/{}/view_1.npy'.format(method))
        self.view_2=np.load('embeds/{}/view_2.npy'.format(method))
        self.view_3=np.load('embeds/{}/view_3.npy'.format(method))
        self.create_finder()
    
    def create_finder(self):
        self.finder_view_1=Neighbor_finder(self.view_1)
        self.finder_view_2=Neighbor_finder(self.view_2)
        self.finder_view_3=Neighbor_finder(self.view_3)
    
    def get_view_among_all(self, finder_view, choose_id, n):
        all_D=[]
        all_I=[]
        D, I = finder_view.get_neighor_by_vector(self.view_1[choose_id],n=n[0]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1])
        print('q_1:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_1:',raw_text[i])

        D, I = finder_view.get_neighor_by_vector(self.view_2[choose_id],n=n[1]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1])
        print('q_2:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_2:', raw_text[i])

        D, I = finder_view.get_neighor_by_vector(self.view_3[choose_id],n=n[2]+1)
        all_D.append(D[0][-1])
        all_I.append(I[0][-1]) 
        print('q_3:',self.raw_text[I[0][-1]])
    #     for i in I[0]:
    #         print('q_3:', raw_text[i])
        print('Choose facet:', np.argmax(all_D)+1)
        print('Max:', self.raw_text[all_I[np.argmax(all_D)]])
        
    
    def vis(self, index, model):
        X=np.concatenate([self.view_1[index],self.view_2[index], self.view_3[index]])
        if model=='tsne':
            X_embedded = TSNE(n_components=2).fit_transform(X)
        else:
            X_embedded = MDS(n_components=2).fit_transform(X)
        X_embedded=X_embedded.reshape(3,-1,2)
        c_map={1:'r', 2:'b',3:'g'}
        for i in c_map:
            plt.scatter(X_embedded[i-1][:,0], X_embedded[i-1][:,1], color=c_map[i], label="facet "+str(i))

        plt.legend(loc="upper left")
        plt.title(self.method)
        plt.show()
    


