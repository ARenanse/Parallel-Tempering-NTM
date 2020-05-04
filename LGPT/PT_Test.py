from LGPT import PTReplicaMetaBase
from PTReplica import BasicModel
import multiprocessing as mp

import copy
import numpy as np

import torch

import ParallelTemperingClass as pt

import time

#DNM:- Does not matter.


if __name__ == "__main__":

	Model = pt.ParallelTempering(BasicModel, 1, 100, 1, 5000, 'GEO')

	time.sleep(2)

	train = torch.tensor([[1,2,3,10],[4,5,6,10],[7,8,9,10],[11,12,13,10],[14,15,16,10]], dtype = torch.float)
	test = torch.tensor([[17,18,19,10],[20,21,22,10],[23,24,25,10]], dtype = torch.float)





	#                         DNM     DNM                                     DNM
	Model.InitReplicas(3,5,1, 200,0.6,700,True,0.5,train,test,0.001,0.025, "ANYTHING")

	#print(Model.ReplicaList[0].Model.state_dict())

	Model.CopyandSetReplicas()

	# Model.RunReplicas()

	# print( len(Model.SamplesListAllReplicas[0]) )

	# Model.SwapExecutor()

	# print(Model.ReplicaList[0].Model.state_dict())