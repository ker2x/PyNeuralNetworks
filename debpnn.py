# Mixed Differential Evolution - Back-Propagation 3-Layer Percepton Neural Network
# 
# Originally Written in Python.  See http://www.python.org/
# Placed in the public domain.
# by Neil Schemenauer <nas@arctrix.com>
# Currenty Version was created by
# Jose Antonio Martin H.
# Placed in the public domain.
#  Added Differential Evolution.
#  Added Saving and Loading of weigths from file
#  Needs for course the  DESolver module (http://www.icsi.berkeley.edu/~storn/code.html)
#  by the people who invented Differential Evolution
# Please enjoy it!.
# Cheers.
# Jose Antonio Martin H



import math
import random
import string
import time
import cPickle
random.seed(math.pi)
import DESolver
#reload(DESolver) # for debugging and development
#import psyco
#psyco.full()

stBest1Exp          = 0
stRand1Exp          = 1
stRandToBest1Exp    = 2
stBest2Exp          = 3
stRand2Exp          = 4
stBest1Bin          = 5
stRand1Bin          = 6
stRandToBest1Bin    = 7
stBest2Bin          = 8
stRand2Bin          = 9
true  = 1
false = 0

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill for i in range(J)])
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function
def dsigmoid(y):
    return 1.0-y*y

class NN(DESolver.DESolver):
    def __init__(self, ni, nh, no,pat):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0 for i in range(self.ni)]
        self.ah = [1.0 for i in range(self.nh)]
        self.ao = [1.0 for i in range(self.no)]
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)


        # Setup Differential Evolution
        self.patterns=pat
        self.dim = int(self.ni*self.nh + self.nh*self.no)
        pop = self.dim+200
        DESolver.DESolver.__init__(self, self.dim, pop) # superclass
        self.count = 0
        self.testGenerations = 100
        self.cutoffEnergy = 0.0

        self.mine =9999999999999999
        min = [None for i in range(self.dim)]
        max = [None for i in range(self.dim)]

        for i in range(self.dim):
            max[i] =  2.0
            min[i] = -2.0

        self.Setup(min,max,stBest1Exp,0.9,1.0)
        self.setCutoffEnergy(0.0)

    def SaveW(self,filename):
         W = [self.wi,self.wo]
         cPickle.dump(W,open(filename,'w'))
         

    def LoadW(self,filename):         
         W = cPickle.load(open(filename,'r'))
         self.wi=W[0]
         self.wo=W[1]
    
    def setCutoffEnergy(self, energy):
        self.cutoffEnergy = energy
        

    def MakeLayer(self,data,I,J):
        #print "Make layer ",len(data),i,j,i*j
        m=[]
        cont=0
        for i in range(I):
            row=[]
            for j in range(J):
                row.append(data[cont])
                cont=cont+1
            m.append(row)
                
                
        return m
    
    def EnergyFunction(self, trial, bAtSolution):

        ninh = trial[0:self.ni*self.nh]
        nhno = trial[self.ni*self.nh:self.dim]
        
        self.wi = self.MakeLayer(ninh,self.ni, self.nh)
        self.wo = self.MakeLayer(nhno,self.nh, self.no)

        result  = self.GetError(self.patterns)
        
        self.count += 1

            
        # self.count is per evaluation, self.count % nPop is per self.generation
        if (self.count-1)%self.nPop == 0:
            self.generation = self.count / self.nPop
            #print self.count, self.nPop, self.count / self.nPop, self.Energy()
           

            # we will be "done" if the energy is less than or equal to the cutoff energy (default 0.0)
            if self.Energy() <= self.cutoffEnergy:
                    bAtSolution = true

            # we will be "done" if the energy is changed by less that 10% every "self.testGenerations" generations
            if self.generation == self.testGenerations: # set initial test energy
                self.testEnergy = self.Energy()

            # test every self.testGenerations generations after the initialization above
            if self.generation > self.testGenerations and self.generation % self.testGenerations == 0:
                
                 #if energy changes by less than 50% in "self.testGenerations" generations, stop
                deltaEnergy = self.testEnergy - self.Energy()
                if deltaEnergy < (self.testEnergy/2.0):
                    bAtSolution = true

                self.testEnergy = self.Energy()
        return result, bAtSolution

    def DETrain(self,iterations):
        self.Solve(iterations)
        solution = self.Solution()

        ninh = solution[0:self.ni*self.nh]
        nhno = solution[self.ni*self.nh:self.dim]
        
        self.wi = self.MakeLayer(ninh,self.ni, self.nh)
        self.wo = self.MakeLayer(nhno,self.nh, self.no)
    
    
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'wrong number of inputs'

        # input activations
        for i in range(self.ni-1):            
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def puntualError(self,targ):
        xerror = float(0.0)       
        for k in range(len(targ)):
            xerror = xerror + 0.5 * ((targ[k]-self.ao[k])**2)
        return xerror
        
    
    def GetError(self,pat):
        xerror = float(0.0)        
        for p in pat:
            inputs  = p[0]
            targets = p[1]
            self.update(inputs)
            xerror = xerror + self.puntualError(targets)
            #xerror = xerror + self.backPropagate(targets, 0.5, 0.1)            
        return xerror
        

    def test(self, patterns):
        for p in patterns:
            print p[0], '->', self.update(p[0])

    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]

    def singletrain(self,input,target):
        self.update(input)
        self.self.backPropagate(targets,0.5, 0.1)

        
    def train(self, patterns, iterations=100, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error  = 0.0            
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                
            error2 =  self.GetError(patterns)
            if i % 100 == 0:
                print 'error1 %-14f ' % error , 'error2 %-14f ' % error2


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
    n = NN(2, 3, 1,pat)
    #train it with some patterns
    print "Staring backpropagation batch training"
    n.train(pat,1000)  # Train is with Back Propagation Algorithm
    # test it
    n.test(pat)
    b=time.clock()
    print "Total time for Back Propagation Trainning ",b-a
    n.SaveW("NN.dat")  # Save Weigths to file
    del n
    n = NN(2, 3, 1,pat)
    n.LoadW("NN.dat")  # Load Weigths from file
    n.test(pat)
    print
    print
    del n
    a = time.clock()
    n = NN(2, 3, 1,pat)
    print "Staring Differential Evolution Trainning Algorithm"
    n.DETrain(15)     # Evolutionary Differential Evolution Trainning Algorithm
    print "error",n.GetError(pat)
    n.test(pat)
    b=time.clock()
    print "Total time for Differential Evolution Trainning ",b-a
    

if __name__ == '__main__':
    demo()
