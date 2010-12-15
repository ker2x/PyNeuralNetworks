from Numeric import *
from RandomArray import *
import time



class SOM:
    def __init__(self,size_I=10,size_J=20,size_K=10):
        seed(3,3)
        self.I= size_I
        self.J= size_J
        self.K= size_K
        self.W= uniform(-10,10,(self.I,self.J,self.K))
        #self.W= zeros((self.I,self.J,self.K),Float)
        self.Y= sum(self.W,2)
        self.T= 0.0
        self.Alpha0=0.9
        self.Alpha=self.Alpha0        
        self.N_Iterations=1
        self.H= zeros((self.I,self.J,self.K))
        self.i_min=0
        self.j_min=0
        self.min_val=9999999999999999
        self.Ratio0=1.0
        self.Ratio0=max(self.J,self.I)
        #self.Ratio0=0.0        
        self.Ratio=self.Ratio0
        


    def alpha(self):
        #Ritmo de Aprendizaje
        self.Alpha=self.Alpha0+(0.01-self.Alpha0) * ( self.T / self.N_Iterations)        
        return self.Alpha
    
    def R(self):
        #Radio de Vecindad
        self.Ratio=self.Ratio0+(1.0-self.Ratio0)*( self.T / self.N_Iterations)       
        return self.Ratio    
    
    def H_i_g(self):
        for i in range(self.I):
            for j in range(self.J):
                d=sqrt( ((i-self.i_min)**2) + ((j-self.j_min)**2) )
                if d > self.R():
                    self.H[i,j,:]=0
                else: 
                    self.H[i,j,:]=1
               

    def NeuroWinner(self):
        #Determinar la Neurona Ganadora
        self.min_val=9999999999999999999999
        self.i_min=0
        self.j_min=0
        for i in arange(self.I):
            for j in arange(self.J):              
               temp_val= self.Y[i,j]
               if temp_val < self.min_val:
                   self.i_min =i
                   self.j_min=j
                   self.min_val=temp_val


        
    def Propagate(self,X):
        #Calcular la Distancia Euclidea = Propagar el estimulo
        self.Y=sum(abs(self.W-X),2)
        #print self.Y
        #Determinar la Neurona Ganadora
        self.NeuroWinner()
        
    def Learn(self,X):
        #Actulizar la matriz H
        self.H_i_g()        
        #Actualizar Pesos        
        self.W=self.W+ self.alpha() * self.H * ( X - self.W)               
        
    def Print(self,X):
        #print "Entrada:                   " ,X
        print "La Neurona Ganadora es:    " ,[self.i_min,self.j_min]
        print "Con Vector Caracteristico: " , self.W[self.i_min,self.j_min,]
        #print "Ratio,T,Alpha",self.Ratio,self.T,self.Alpha
                   
    def Train(self,X,N=1000):
        # X es un array de vectores de entrenamiento
        self.N_Iterations=N
        num_samples_vectors=X.shape[0]
        for i in range(N):
            self.T+=1.0
            #print "iteration #:",i          
            #print "Ratio,T,Alpha",self.Ratio,self.T,self.Alpha
            for j in range(num_samples_vectors):
                #Propagate(X[j])
                self.Y=sum(abs(self.W-X[j]),2)
                self.NeuroWinner()
                #Learn(X[j])
                self.H_i_g()        
                #Actualizar Pesos        
                self.W=self.W + self.alpha() * self.H * (X[j] - self.W)
                
    def ClasifyPattern(self,X):
        self.Propagate(X)
        self.Print(X)
        
    

def PruebaSOM():
    red=SOM(4,4,5)
    Entrada=array([[0,0,0,0,0],
                   [64,64,64,64,64],
                   [128,128,128,128,128],
                   [255,255,255,255,255],             
                   ])
    
    t1=time.clock()
    red.Train(Entrada,1000)  
    t2=time.clock()
    print "El tiempo es: ",t2-t1
    print "Patrones Orinigales"
    for i in arange(Entrada.shape[0]):
        red.ClasifyPattern(Entrada[i])

    print
    print
    Prueba=array([ [3,3,3,3,3],
                   [61,61,61,61,61],
                   [120,120,120,120,120],
                   [275,275,275,275,275],             
                   ])
    for i in arange(Prueba.shape[0]):
        red.ClasifyPattern(Prueba[i])
   


if __name__ == '__main__':
    PruebaSOM()
   
