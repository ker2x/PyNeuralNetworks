#from __future__ import division
from Numeric import *
#from MA import *
from RandomArray import *
import time
from pylab import * # matplotlib



class SOM:
    def __init__(self,size_I=10,size_J=20,size_K=10):
        #seed(3,3)
        
        self.I            = size_I
        self.J            = size_J
        self.K            = size_K
        #mascara           = randint(-2,2,(self.I,self.J,self.K))
        #mascara           = where (mascara <=0,0,1)
        #self.W            = array(uniform(-1000,1000,(self.I,self.J,self.K)),mask=mascara )
        self.W            = uniform(-1000,1000,(self.I,self.J,self.K))
        
        # array of standar deviations
        #self.S            = uniform(10.0,20,(self.I,self.J,self.K)).astype(float)
        self.S            = ones((self.I,self.J,self.K)).astype(float)+20.0        
        self.Y            = zeros((self.I,self.J)).astype(float)
        #self.Generality  = sum(self.W.mask(),2)
        self.T            = 0.0
               
        self.N_Iterations = 1

        # Matrix for calculating the neigborhood
        self.H            = zeros((self.I,self.J,self.K))

        # Matrix for calculating the neigborhood for the stdev
        self.HS           = zeros((self.I,self.J,self.K)).astype(float)
        
        #Winer Neuron
        self.i_min        = 0
        self.j_min        = 0
        self.activation   = 0

        # factor or learning
        self.Alpha0       = 0.9
        self.Alpha        = self.Alpha0 

        # factor or neigborhood
        self.Ratio0       = sqrt( self.J**2 + self.I**2 )        
        self.Ratio        = self.Ratio0
        
    def Gauss(self,mu,s,x):
        return exp ( - ((x-mu)/s)**2.0 )

    def alpha(self):
        #Learning Rate
        self.Alpha = self.Alpha0 *  ( 1.0 - ( float(self.T) / self.N_Iterations) )       
        return self.Alpha
    
    def R(self):
        #Neigbourhood ratio
        self.Ratio = self.Ratio0 *  ( 1.0 - ( float(self.T) / self.N_Iterations) )
        return self.Ratio

    def dist(self,i,j,k):
        return  sqrt( ((i-self.i_min)**2) + ((j-self.j_min)**2) )


    def distGauss(self,i,j,k):
        R=self.R()
        if R<1: R=1.0        
        x= self.Gauss( self.i_min, R ,i )
        y= self.Gauss( self.j_min, R ,j )
        return x*y
    
    def H_i_g(self):
        # Actualize Neigbourhood ratio
        #self.H = fromfunction( self.dist , (self.I,self.J,self.K))
        #R = self.R()                
        #self.H = where (self.H <= R,1.0/(1.0+tanh(self.H)),0)
        self.H  = fromfunction( self.distGauss , (self.I,self.J,self.K))
        self.HS = where (self.H > 0,self.H,1)
        
    def NeuroWinner(self):
        #find the winning neuron
        pos              = argmax(self.Y.flat)
        #pos             = argmin(self.Y.flat)
        #print "max,min",max(self.Y.flat),min(self.Y.flat)
        self.i_min      = pos / self.J
        self.j_min      = pos % self.J                
        #self.activation = 1.0/ (1+self.Y[self.i_min,self.j_min])
        self.activation = self.Y[self.i_min,self.j_min]
        

        
    def Propagate(self,X):        
        #Compute euclidian distance
        #self.Y  = sqrt(sum((self.W-X)*(self.W-X),2))
        self.Y  = product(self.Gauss(self.W,self.S,X),2)        
        #self.Y  = average(self.Gauss(self.W,self.S,X),2)
        
        
        #find the winning neuron
        self.NeuroWinner()
        
    def Learn(self,X):
        #Actulize the matrix H
        self.H_i_g()        
        #Actualize Weight        
        self.W=self.W + self.alpha() * self.H * ( X - self.W)               
        #Actualize deviation
        #self.S = self.S * self.H
        #self.S[self.i_min,self.j_min]=self.S[self.i_min,self.j_min]+ 2*self.Y[self.i_min,self.j_min]
        #self.S = where(self.S<1.0,1,self.S)
        # Evolution
        # Mutation de la mascara
        # Mutacion de valores
        # Cruzamiento de nodos  (valores y mascaras)
        
        
    def Print(self,X):
        #print "Entrada:                   " ,X
        print "the winning neuron is :    " ,[self.i_min,self.j_min]
        print "with vector : " , self.W[self.i_min,self.j_min,]
        #print "Ratio,T,Alpha",self.Ratio,self.T,self.Alpha
                   
    def Train(self,X,N=1000):
        # X  is an array of training vector
        self.N_Iterations   = N
        num_samples_vectors = X.shape[0]
        self.T              = 0
        for i in range(N):
            self.T+=1.0            
            for j in range(num_samples_vectors):
                self.Propagate(X[j])                
                self.Learn(X[j])
                
                
    def ClasifyPattern(self,X):
        self.Propagate(X)        
        return array([self.i_min,self.j_min])



def SOP(nxgrid,nygrid,Xinput,Niter=100):
    clases = zeros((Xinput.shape[0],2)).astype(float)
    centroids = zeros((Xinput.shape[0],Xinput.shape[1])).astype(float)
    activations = zeros(Xinput.shape[0]).astype(float)
    NET=SOM(nxgrid,nygrid,Xinput.shape[1])
    NET.Train(Xinput,Niter)
    for i in range(Xinput.shape[0]):
        c  = NET.ClasifyPattern(Xinput[i])
        clases[i,0]    = c[0]
        clases[i,1]    = c[1]
        centroids[i,:] = NET.W[c[0],c[1]]
        activations[i] = NET.activation
    return clases,centroids,activations


def PlotAct(ZMaps,vectors):    
    
    figure(1)

    N=ZMaps[0].shape[0]
    M=ZMaps[0].shape[1]
    ZTotal=ZMaps[0]
    for i in range(len(ZMaps)):        
        subplot(len(ZMaps)/2,2,i+1)
        #im = imshow(-ZMaps[i], interpolation='bilinear', origin='lower',cmap=cm.gray, extent=(0,N,0,M))
        im = imshow(-ZMaps[i],interpolation='spline36', origin='lower',cmap=cm.gray, extent=(0,N,0,M))
        hot()
        #axis('off')
        ZTotal=maximum(ZTotal,ZMaps[i])
        #colorbar()
        #title('vector: '+str(vectors[i]))

    figure(2)
    im = imshow(-ZTotal,interpolation='spline36', origin='lower',cmap=cm.gray, extent=(0,N,0,M))
    hot()
    show()


def PruebaSOM2():
    
    Entrada=array([
                   [-25,-25,-25,-25],
                   [-26,-26,-26,-26],
                   [0,0,0,0],
                   [1,1,1,1],                   
                   [67,67,67,67],
                   [69,69,69,69],
                   [128,128,128,128],
                   [127,127,127,127],
                   [255,255,255,255],
                   [253,253,256,257],
                   ]).astype(float)

   
    ActMaps=[]
    vectors=[]
    clases = zeros((Entrada.shape[0],2)).astype(float)
    centroids = zeros((Entrada.shape[0],Entrada.shape[1])).astype(float)
    activations = zeros(Entrada.shape[0]).astype(float)
    NET=SOM(40,40,Entrada.shape[1])
    NET.Train(Entrada,100)
    for i in range(Entrada.shape[0]):
        c  = NET.ClasifyPattern(Entrada[i])
        clases[i,0]    = c[0]
        clases[i,1]    = c[1]
        centroids[i,:] = NET.W[c[0],c[1]]
        activations[i] = NET.activation         
        print Entrada[i], clases[i,:],centroids[i,:],activations[i]
        ActMaps.append(NET.Y)
        vectors.append(Entrada[i])
           

    PlotAct(ActMaps,vectors)
    
    

def PruebaSOM1():
    
    Entrada=array([
                   [-25,-25,-25,-25],
                   [-26,-26,-26,-26],
                   [0,0,0,0],
                   [1,1,1,1],
                   [35,35,35,1],
                   [67,67,67,67],
                   [69,69,69,69],
                   [128,128,128,128],
                   [127,127,127,127],
                   [255,255,255,255],
                   [253,253,256,257],
                   ]).astype(float)

    



    t1=time.clock()
    clases,centroids,act = SOP(10,10,Entrada,100)    
    t2=time.clock()
    print "SOM: trining time :",t2-t1
    
    
    for i in range(Entrada.shape[0]):      
      print Entrada[i], clases[i,:],centroids[i,:],act[i]


    
    

if __name__ == '__main__':
    PruebaSOM2()
   
