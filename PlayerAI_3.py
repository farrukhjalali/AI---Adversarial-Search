# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:42:41 2017

@author: Farrukh Jalali

Implementing Player AI logic here. Using the huesritstics and alpha beta with pruning adverserial serach. Two heuristics are ultimately used; Cornerness and Highest number possible present on the board. Weights for cornerness are set as w1 for after optimizing with brute force optimized alogrithm.
"""

from BaseAI import BaseAI
import copy
import numpy as np
import scipy.stats as ss

import Grid_3

DEPTH = 3

w1 = np.array([1,0.525,0.05,0.01, 0.52,0.12,0.01,0, 0.05,0.01,0.01,0, 0,0,0,0])
w2 = np.array([1,0.01,0.0,0,0.01,0,0,0,0,0,0,0,0,0,0,0])

w3 = np.array([15, 13, 10,  6, 14, 11,  7,  3, 12,  8,  4,  1,  9,  5,  2,  0])

w4 = np.array([15, 14, 12,  9, 13, 11,  8,  5, 10,  7,  4,  2,  6,  3,  1,  0])


class PlayerAI(BaseAI):
    def getMove(self, grid):
        d = DEPTH
        alpha = {'move':-1, 'value':float("-inf")}
        beta = {'move':-1, 'value':float("inf")}
        
        if grid.canMove():
            bestHeuristic =  self.alphaBetaAlgo(grid, alpha, beta, d, grid.map, None)
#            print("returning finally move: %s" % bestHeuristic['move'])
            return bestHeuristic['move']
        else:
            return None
        
        #return moves[randint(0, len(moves) - 1)] if moves else None
    def getEvaluationScore(self, grid, orig, move):
        maxVal = grid.getMaxTile()
        
        #map1 = np.argsort(np.ravel(grid.map))
        flatGrid = np.ravel(grid.map)
        
        Cornerness = (flatGrid * w1)
        Cornerness[np.isneginf(Cornerness)] = 1
        Cornerness[Cornerness==0] = 1
        Cornerness = np.log2(sum(Cornerness))
                
        HighNumber = np.log2(sum(maxVal * w2))
        """
        moveness = 1
        
        if move == 0:
            moveness = 1.75
        elif move == 1:
            moveness = 0.3
        elif move == 2:
            moveness = 1.2
        """    
        #order = np.argsort(flatGrid)
        #Orderness = -1*sum(map(sum,np.diff(grid.map, axis=0))) + sum(map(sum,np.diff(grid.map, axis=1)))
        #abs_Orderness = max(abs(Orderness),1)
        
        #s_max = np.log2(maxVal)-1
        #Emptyness = np.log2((flatGrid==0).sum())
        
        #Highnumberness = np.log2(sum(map(sum,grid.map)))
        #Unmergedness = unmergedScore
        #Emptyness = np.log2(zerosScore)
        #Orderness = np.log2(Orderness)
        #print("zerosScore : %d" % zerosScore)
        #print("unmergedScore : %d" % unmergedScore)
        return 1.5*Cornerness + HighNumber #+ moveness + (Orderness/abs_Orderness*np.log2(abs_Orderness))
  
    def alphaBetaAlgo(self, grid, alpha, beta, d, Orig, move, MrMaxPlaying=True):
        if d == 0 or not grid.canMove():
            val = self.getEvaluationScore(grid, Orig, move)
            #print("returning move : %d with heuristic %f" % (move,val))
            return {'move': move, 'value':val}

        if MrMaxPlaying:
            bestValue = copy.deepcopy(alpha)
            
            moves = grid.getAvailableMoves()
            
            for m in moves:
                cloneGrid = grid.clone()
                cloneGrid.move(m)
                #print("depth: %d, move: %d, is max: %d" %(d, m, MrMaxPlaying))
                if d == DEPTH:
                    move = m                   
                newAlpha = self.alphaBetaAlgo(cloneGrid,bestValue, beta,  d-1, Orig, move, False)
                #print("newAlpha value: %d"%newAlpha['value'])
                if bestValue['value'] == float('-inf'):
                    t = -11111111
                else:
                    t = bestValue['value']
                #print("selecting maximum between bestValue %f and newalpha %f"%(t,newAlpha['value']))
                
                bestValue = max(bestValue, newAlpha, key=lambda x: x['value'])
                #print("bestValue for maxi is now %f with move %d at depth %d"%(bestValue['value'],bestValue['move'],d))
                    
                if bestValue['value'] >= beta['value']:
                #    print("alpha %f < beta %f in mr max of depth %d with value %f "%(bestValue['value'],beta['value'],d,bestValue['value']))
                    break
        else:
            bestValue = copy.deepcopy(beta)
            
            moves = grid.getAvailableMoves()
            
            for m in moves:
                cloneGrid = grid.clone()
                cloneGrid.move(m)
                #print("depth: %d, move: %d, is min: %d" %(d, m,MrMaxPlaying))
                if d == DEPTH:
                    move = m                   
                newBeta = self.alphaBetaAlgo(cloneGrid, alpha, bestValue, d-1, Orig, move, True)                
                if bestValue['value'] == float('inf'):
                    t = 11111111
                else:
                    t = bestValue['value']
                #print("selecting minimum between bestValue %f and newbeta %f "%(t,newBeta['value']))
                
                bestValue = min(bestValue, newBeta, key=lambda x:x['value'])
                #print("bestValue for mini is now %f with move %d at depth %d"%(bestValue['value'],bestValue['move'],d))
                
                if bestValue['value'] <= alpha['value']:
                #    print("alpha %f < beta %f in mr min of depth %d with value %f "%(alpha['value'],bestValue['value'],d,bestValue['value']))
                    break
        return bestValue
"""
g = Grid()
g.map = [[2,64,32,8], [4,32,8,2], [32,4,2,4],[4,2,4,2]]
g.map = [[2,2,2,2], [4,2,2,2], [2,4,2,4],[4,2,4,2]]

g.map
y = np.log2(g.map)
x = np.ravel(g.map)
x

np.argsort(x)[14]

order = np.array(ss.rankdata(x))
order
w3-order
np.log2(sum(abs(w3-order)))
(w4-order)
np.log2(sum(abs(w4-order)))

sorted(range(len(x)),key=lambda y:x[y])
y
sum(map(sum,np.diff(g.map, axis=0)))
sum(map(sum,np.diff(g.map, axis=1)))
"""



"""
        unmergedScore = 0
        zerosScore = 0
        maxVal = grid.getMaxTile()
        size = grid.size
        
        avg = np.mean(grid.map[grid.map!=0])
        med = np.median(grid.map)
        
        map1 = np.argsort(np.ravel(grid.map))
        bmark = np.array(range(size*size))
        Order = np.array(bmark - map1)
        Disorderness = sum(abs(Order))/size * -2
        
        MaxCornerness = 0
        pos = np.array((3,3))
        for p in pos:
            if grid.getCellValue(pos) == maxVal:
                MaxCornerness += 3
            
        zeros = (grid.map==np.zeros((4,4))).astype(int)
        oZeros = (orig==np.zeros((4,4))).astype(int)
        mergeness = sum(map(sum,zeros - oZeros))
        
        zerosScore = sum(map(sum,zeros))
        
        logMap = np.log2(grid.map)
        logMap[np.isneginf(logMap)] = 0

        Highnumberness = sum(map(sum,logMap))
        #Unmergedness = unmergedScore * -200
        Emptyness = zerosScore
        #print("zerosScore : %d, Disorderness: %d, Highnumberness : %d, Emptyness : %d, Maxcornerness : %d" % (zerosScore, Disorderness, Highnumberness, Emptyness, MaxCornerness))
#        print("grid now: " )
#        print(grid.map)
        #print("orig:")
        #print(orig)
        #print("total : %d" % total )
        t = 2*avg+4*med+7*Emptyness
#        print("total : %d" % t)
        return t
"""