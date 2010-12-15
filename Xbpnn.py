# Back-Propagation Neural Networks
# 
# Written in Python with Numeric python 
# 
# Jose Antonio Martin H <jamartin AT dia fi upm es>
from Numeric import *
from RandomArray import *
import cPickle
from numpy import * # use this if you use numpy instead of Numeric
import time

seed(3,3)

class NN:

    def sigmoid(self,x):
        return tanh(x)

# derivative of our sigmoid function
    def dsigmoid(self,y):
        return 1.0-y*y

    
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = ones((self.ni),Float)
        self.ah = ones((self.nh),Float)
        self.ao = ones((self.no),Float)
        
        # create weights
        self.wi = uniform(-2.0,2.0,(self.ni, self.nh))
        self.wo = uniform(-2.0,2.0,(self.nh, self.no))
        

        # last change in weights for momentum   
        self.ci = zeros((self.ni, self.nh),Float)
        self.co = zeros((self.nh, self.no),Float)

    def SaveW(self,filename):
         W = [self.wi,self.wo]
         cPickle.dump(W,open(filename,'w'))
         

    def LoadW(self,filename):         
         W = cPickle.load(open(filename,'r'))
         self.wi=W[0]
         self.wo=W[1]

    def update(self, inputs):
        if len(inputs) != self.ni-1: 
            raise ValueError, 'wrong number of inputs'

        # input activations
        self.ai[0:self.ni-1]=inputs

       
        # hidden activations
        sum = matrixmultiply(transpose(self.wi),self.ai)
        self.ah = tanh(sum)
        
        # output activations
        sum = matrixmultiply(transpose(self.wo),self.ah)
        self.ao = tanh(sum)           


        return self.ao


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas =  self.dsigmoid(self.ao) * (targets-self.ao)


        # calculate error terms for hidden
        error = matrixmultiply(self.wo,output_deltas)
        hidden_deltas =  self.dsigmoid(self.ah) * error
        
        # update output weights
        change = output_deltas * reshape(self.ah,(self.ah.shape[0],1)) 
        self.wo = self.wo + N  * change + M * self.co
        self.co = change
        

        # update input weights
        change = hidden_deltas * reshape(self.ai,(self.ai.shape[0],1))                 
        self.wi = self.wi + N*change + M*self.ci
        self.ci = change


        # calculate error        
        error = sum(0.5* (targets-self.ao)**2)        
        
        return error


    def test(self, patterns):
        for p in patterns:
            print p[0], '->', self.update(p[0])


    def singletrain(self,inputs,targets):
        self.update(inputs)
        return self.backPropagate(targets,0.5, 0.1)
         

        
    def train(self, patterns, iterations=100, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)                
            if i % 100 == 0 and i!=0:
                print 'error ' + str(error)


    
def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [-1]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [-1]]
    ]

    # create a network with two input, two hidden, and one output nodes
    a = time.clock()
    n = NN(2, 3, 1)
    #train it with some patterns
    print "Starting bath training"
    n.train(pat,1000)  # Train is with Back Propagation Algorithm
    # test it
    n.test(pat)
    b=time.clock()
    print "Total time for Back Propagation Trainning ",b-a
    print
    print "Writing Network to file NN.dat"
    n.SaveW("NN.dat")  # Save Weigths to file
    del n
    n = NN(2, 3, 1)
    print "Load network from file NN.dat"
    n.LoadW("NN.dat")  # Load Weigths from file
    n.test(pat)
    del n

    # create a network with two input, two hidden, and one output nodes
    a = time.clock()
    n = NN(2, 3, 1)
    #train it with some patterns
    print "Starting single step training"
    for i in xrange(1000):
            error = 0.0
            for p in pat:
                inputs = p[0]
                targets = p[1]                
                error = error + n.singletrain(inputs,targets) 
            if i % 100 == 0 and i!=0:
                print 'error ' + str(error)
    # test it
    n.test(pat)
    b=time.clock()
    print "Total time for Back Propagation Trainning ",b-a
    print
    print "Writing Network to file NN.dat"
    n.SaveW("NN.dat")  # Save Weigths to file
    del n
    n = NN(2, 3, 1)
    print "Load network from file NN.dat"
    n.LoadW("NN.dat")  # Load Weigths from file
    n.test(pat)
    del n


    

if __name__ == '__main__':
    demo()




