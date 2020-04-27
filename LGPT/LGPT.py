import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import multiprocessing
import torch

#Uses CamelCase all around.

#Some Notations:-
#  1. ?? ---> Shape not known yet.
#  2. ? ---> Type of the variable Not understood yet.
#  3. Shape of the data is assumed to be of the form [BatchSize, Timesteps, Features] for general purpose use across various tasks.
#     Therefore, if one wishes to pass a Regression or Classification Task, then the shape would be [BatchSize, 1, Features]
#  4. Space in subsequent lines of Docstring is used to distinguish between the parameters which are still fuzzy (due to 1 or 2 or both) and which are not.
#  5. !! ---> Projected to use in a feature in future updates.



class PTReplicaMetaBase(ABC):
    
    def __init__(self, Model, NumSamples, GlobalFraction, Temperature, UseLG, LGProb, TrainData, TestData, lr, RWStepSize, Optimizer = torch.optim.SGD, LossFunc = torch.nn.MSELoss, ):
        
        """
        Model : (?) Pytorch Model.
        
        NumSamples : (int), No. of samples to find for this Replica.
        GlobalFraction : (float), Fraction of NumSamples in which Temperature will be assigned as per the Beta scheme.
        Temperature : (float), Temperature assignment for this Replica.
        UseLG : (bool), Whether to use Langevin Gradients or not.
        LGProb : (float), Probability by which to choose Langevin Dynamics for MH proposal distributions.
        
        TrainData : (NP Array?) [BatchSize, Timesteps, Features], Data to train the model on.
        TestData : (NP Array?) [BatchSize, Timesteps, Features], Data to test the model on for validation error trace.

        lr : (float), Learning Rate.
        RWStepSize : (float), Step Size for Random Walk.
!! ---> Optimizer : torch.optim 's Method, the optimizer to use while evaluating Langevin Gradients, default as SGD.
        LossFunc : torch.nn 's Method, the Loss function to use while evaluating Langevin Gradients, used in self.GiveMeTheLoss
       
        
        """
        super().__init__()
        
        self.Model = Model
        
        self.NumSamples = NumSamples
        self.GlobalFraction = GlobalFraction
        self.Temperature = Temperature
        self.UseLG = UseLG
        self.LGProb = LGProb
        self.TrainData = TrainData
        self.TestData = TestData
        
        self.GlobalSamples = NumSamples*GlobalFraction
        
        self.Optimizer = Optimizer(self.Model.parameters(),lr = lr)
        self.LossFunc = LossFunc(reduction = 'mean')
        self.learning_rate = lr
        self.RWStepSize = RWStepSize
        
        #Other Class Related Variables
        self.accepts = 0
        self.ReplicaBeta = None  #The inverse of Temperature for this Replica, used in Likelihod and prior calculation.
    
    
    @abstractmethod
    def PriorLikelihood(self):
        """
        Calculates the Prior Log Likelihood of the Model parameters as per the Prior distribution.
        Should Return the log probability summed over all Weight and Biases.
        
        Returns an info list too if there are measures that one might need to track.

        """
        print("Abstract Function without any implementation called!!!")
        
        pass
    
    
        
    @abstractmethod
    def Likelihood(self):
        """
        Calculates the Log Likelihood over all the instances in Data Train 
        according to the Likelihood Distribution you choose to decide/implement after inheriting this class.
        
        Returns an info list too if there are measures that one might need to track.
        """
        print("Abstract Function without any implementation called!!!")

        pass
    
                
        
    @abstractmethod
    def GiveMeTheLoss(self):
        """
        Returns the loss using the self.LossFunc AFTER computing y_pred from Model as desired.
        
        Abstracting this because calculating y_pred becomes 'Model' and 'TrainData' specefic task.
        """
        print("Abstract Function without any implementation called!!!")
        
        pass
    
    
        
    @abstractmethod
    def ProposeMiscParameters(self):
        """
        Proposes new values to those parameters (will be called Miscelaneous parameters) which are used
        in calculation of PriorLikelihood and/or in Likelihood by calling it in self.Runner .
        
        It returns new proposed values for Miscellaneous Parameters.
                
        RETRUNS:
        
        New values for the Miscellaneous Parameters in a list, so the order is important. 
        """
        print("Abstract Function without any implementation called!!!")
        
        pass
    
        
        
    
    def __ParamClonetoList(self):
        
        """
        Returns a list of model parameters' COPY/CLONE.
        """
    
        ClonedParams = []
        
        for param in self.Model.parameters():
            ClonedParams.append(param.clone())

        return ClonedParams
    
    def _ParamClonetoDict(self):
        
        """
        Returns a dict of model parameters' COPY/CLONE with the identical keys as the model's.
        """
        
        keys = list(self.Model.state_dict().keys())
        
        return dict(zip( keys, self.__ParamClonetoList() ))
    
    
    def __NonLinCombLists(self, a, List1, pow1, b, List2, pow2):
        
        """
        Calculates and Returns: a*List1**pow1 + b*List2**pow2
        
        Each element of the List is a Tensor.
        """
        
        with torch.no_grad():

            lenList1 = len(List1)
            assert lenList1 == len(List2)

            result = [0 for _ in range(lenList1)]

            for i in range(lenList1):
                result[i] = a * (List1[i].clone()**pow1) + b * (List2[i].clone()**pow2)

        return result
    
    
    def __ZeroTensorListLike(self, this):
        
        """
        Returns a list conatining Tensors filled with zero with shapes exactly like 'this' members' shape.
        """
        result = []
        for param in this:
            result.append(torch.zeros_like(param))
            
        return result
    
    
    
    def __ReduceSumEachElement(self, ParamList):
        
        """
        Calculates aggregate sum of each Tensor in the list and returns that scalar Tensor.
        """
        with torch.no_grad():
            result = 0
            for param in ParamList:
                result += torch.sum(param)

        return result
            
       
    
    def Runner(self):
        
        """
        Runs this Replica for NumSamples according to the LGPT Algorithm to achieve NumSamples from the Posterior Distribution.
        """
        
        self.accepts = 0
        
        CurrentPrior = self.CurrentPriorProb
        CurrentLikelihood = self.CurrentLikelihoodProb
        
        Samples = []
        
        
        ThetaDict = self._ParamClonetoDict()
        
        for i in range(self.NumSamples):
            
            
            if (i < self.GlobalSamples): #Use Global Exploration by Setting Temperature
                
                self.ReplicaBeta = 1/self.Temperature
                
            else : #Use Local Exploration via Canonical MCMC
                
                self.ReplicaBeta = 1
                
            
            #Drawing a sample from U(0,1) to switch between LG Dynamics and Random Walk
            l = np.random.uniform(0,1)
            
            #Let's make a copy of current model parameters as a list as it will be used later.
            ParamCopyList = self.__ParamClonetoList()
            ParamCopyDict = self._ParamClonetoDict()
            
            if ((self.UseLG is True) and (l < self.LGProb)):
                print("I'm in LG!!")
            #PERFORMS LANGEVIN GRADIENT UPDATES for Prior (log)Likelihood and the (log)Likelihood
            
            #Calculating theta_gd = theta_init + alpha*gradient_{theta-init} [ Loss(f_{theta_init}) ]
            #So we need pytorch to calculate gradient of model parameters wrt current parameters set as current model parameters
                    
                #Step 1: Make a copy of current model parameters as a List 
                #----------->Already done.
                #Step 2: Do a backward pass to obtain gradients
                loss = self.GiveMeTheLoss()
                self.Model.zero_grad()
                loss.backward()
                GradsList = []
                for param in self.Model.parameters():
                    GradsList.append(param.grad.data)
                #Step 3: Calculate Theta_gd
                lr = self.learning_rate
                Theta_gd = self.__NonLinCombLists(1, ParamCopyList, 1, -lr, GradsList, 1)
                
            #Calculating Theta_proposal = Theta_gd + N(0, step*I)
                RandList = []
                for theta in Theta_gd:
                    RandList.append(torch.tensor(np.random.normal(0, self.RWStepSize, theta.shape)))
                
                Theta_proposal = self.__NonLinCombLists(1, Theta_gd, 1, 1, RandList, 1)
                
            #Calculate Theta_proposal_gd = Theta_proposal + alpha*gradient_{theta_proposal} [ Loss(f_{theta_proposal}) ]
                
                #Step 1: Set Model Parameters as Theta_proposal
                ProposalStateDict = dict(zip(list(self.Model.state_dict().keys()), Theta_proposal))
                self.Model.load_state_dict(ProposalStateDict)
                
                #Step 2: Do a backward pass to obtain gradients of model parameters wrt to Theta_proposal
                loss2 = self.GiveMeTheLoss()
                self.Model.zero_grad()
                loss2.backward()
                GradsList2 = []
                for param in self.Model.parameters():
                    GradsList2.append(param.grad.data)
                Theta_proposal_gd = self.__NonLinCombLists(1, Theta_proposal, 1, -lr, GradsList2, 1)
                
                #Step 3: Reset the weights of the model to the original for this iteration.
                self.Model.load_state_dict(ParamCopyDict)
                
            #Calculate differences in Current and Proposed Parameters
                
                ThetaC_delta = self.__NonLinCombLists(1, ParamCopyList, 1, -1, Theta_proposal_gd, 1)
                ThetaP_delta = self.__NonLinCombLists(1, Theta_proposal, 1, -1, Theta_gd, 1)
                
                
                
            #Calculate Delta Proposal which is used in MH Prob calculation, note it's delta(differnece) cause we are computing Log Probability for MH Prob
            
                coefficient = 1 / (self.Temperature * 2 * (self.RWStepSize))
                DeltaProposal_List = self.__NonLinCombLists( coefficient, ThetaP_delta, 2, coefficient, ThetaC_delta, 2 )   #The objective output!
                
                DeltaProposal = self.__ReduceSumEachElement(DeltaProposal_List)
                
            
            
            else: 
                print("I'm in MH Random Walk!!")
            #PERFORMS RANDOM WALK UPDATES
                
                DeltaProposal = 0
                
                RandList = []
                for param in ParamCopyList:
                    RandList.append(torch.tensor(np.random.normal(0, self.RWStepSize, param.shape)))
                    
                Theta_proposal = self.__NonLinCombLists(1, ParamCopyList, 1, 1, RandList, 1)
                
                
            #Propose new values to Miscellaneous Parameters using ProposeMiscParameters
            MiscProposalList = self.ProposeMiscParameters()
            
            
            #Calculate Likelihood Probability with the Theta_proposal and New Proposals for Miscellaneous Parameters.(Note this is a log probability)
            LHProposalProb, infoLH = self.Likelihood(MiscProposalList, Theta_proposal)
            print("Likelihood Loss on the Proposed Parameters: ", infoLH)
            #Calculate Prior Probability with the New Proposals for Misc Parameters and/or/maybe the Theta_Proposal too( and if that happens, it implies
            # that calculation of the prior is also dependent on the model which is a highly unlikely case.). 
            #  Note this is a log probability.
            PriorProposalProb, infoPrior = self.PriorLikelihood(MiscProposalList, Theta_proposal)
            
            
            #Calculate DeltaPrior and DeltaLikelihood for MH Probability calculation.
            DeltaPrior = PriorProposalProb - self.CurrentPriorProb
            DeltaLikelihood = LHProposalProb - self.CurrentLikelihoodProb 
            
            #Calculate Metropolis-Hastings Acceptance Probability.
            print("DeltaPrior: ", DeltaPrior)
            print("DeltaLikelihood: ", DeltaLikelihood)
            print("DeltaProposal: ", DeltaProposal)
            
            
            alpha = min(1, torch.exp(DeltaPrior + DeltaLikelihood + DeltaProposal)) 
            print("Alpha: ", alpha)
            
            #EXECUTING METROPOLIS HASTINGS ACCEPTANCE CRITERION
            
            #Draw u ~ Unif(0,1)
            u = np.random.uniform(0,1)
            
            if u < alpha:
                print("Accepted!!")
                print("\n\n")

                
                #Change current Likelihood and Prior Probability.
                self.CurrentLikelihoodProb = LHProposalProb
                self.CurrentPriorProb = PriorProposalProb
                ThetaDict = dict(zip(list(self.Model.state_dict().keys()), Theta_proposal))
                
                #Load The accepted parameters to the model
                self.Model.load_state_dict(ThetaDict)
                
                #Accept the Miscellaneous Parameters
                self.MiscParamList = MiscProposalList
                
                self.accepts += 1
                
                Samples.append(ThetaDict)
                
            else :
                print("Rejected!!")
                print("\n\n")
                
                #Reject all proposals.
                #i.e. Model Parameters remains the same.
                
                Samples.append(ParamCopyDict)
                
    
        
        return Samples, self.accepts
                
                
                
                
                
            
            