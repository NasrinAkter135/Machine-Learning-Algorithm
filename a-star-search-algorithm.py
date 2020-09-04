import numpy as np
from queue import PriorityQueue
import math
import random

n= int(input("Matrix Size,n : "))
obstacle = (int(input("Percentage of obstacles: "))/100)*(n**2-2)

env = np.zeros(n**2).reshape(n, n)
env[0,0] = 7
env[n-1,n-1] = 8
start=env[0,0]
goal=env[n-1,n-1]
q = PriorityQueue()
while obstacle > 0:
    x=random.randrange(n)
    y=random.randrange(n)
    if not ((x == 0 and y == 0) or (x == n-1 and y == n - 1) or env[x,y] == 1):
        env[x,y] = 1
        obstacle-=1

print(env)

class Node:

    def __init__(self, x, y, parent, cost=0):
        self.x=x
        self.y=y
        self.cost=cost
        self.parent=parent
        
    
    def move_left(self,x,y,gx,gy,cost,env):
        if not((y-1<0) or env[x,y-1]==1):
             y=y-1
             mvalue = self.manhattan(x,y,gx,gy)
             fvalue=mvalue+cost
             q.put((fvalue,x,y))
    def move_right(self,x,y,gx,gy,cost,env):
         if not((y+1>n-1) or env[x,y+1]==1):
             x=x
             y=y+1
             print("moveright")
             mvalue=self.manhattan(x,y,gx,gy)
             fvalue =mvalue+cost
             q.put((fvalue,x,y))  
    def move_up(self,x,y,gx,gy,cost,env):
        if not((x-1<0) or env[x-1,y]==1):
            x=x-1
            print("moveup")
            mvalue=self.manhattan(x,y,gx,gy)
            fvalue=mvalue+cost
            q.put((fvalue,x,y))  
    def move_down(self,x,y,gx,gy,cost,env):
        if not((x+1>n-1)or env[x+1,y]==1):
            y=y
            x=x+1
            print("movedown")
            mvalue=self.manhattan(x,y,gx,gy)
            fvalue=mvalue+cost
            q.put((fvalue,x,y))
            
    def manhattan(self,x,y,gx,gy):
         
        mandistance =(abs(x-gx)) + (abs(y-gy) )
        print("manhattan Using math ", mandistance) 
        return mandistance
        
    def is_goal(self,current,goal):
        isgoal=False
        if current==goal:
             isgoal = True
             return isgoal
        
def a_star_search(env, startx,starty, goal):
    
     fscore=0
     q.put((fscore,startx,starty))
     start_node=Node(startx,starty,None,0)
     current=env[start_node.x,start_node.y]
     isgoal = start_node.is_goal(current,goal)
     if isgoal == True:
         print("Source equals to goals")
     else:
         move=0
         cost=0
         while not isgoal==True:
             goalx=n-1
             goaly=n-1
             start_node.move_left(startx,starty,goalx,goaly,cost,env)
             start_node.move_right(startx,starty,goalx,goaly,cost,env)
             start_node.move_up(startx,starty,goalx,goaly,cost,env)
             start_node.move_down(startx,starty,goalx,goaly,cost,env)
             move=move+1
             mindistancevalue=q.get()
             indexx =mindistancevalue[1]
             indexy =mindistancevalue[2]
             env[indexx,indexy]="5"
             print(env)
             print("actualmovement")
             print(move)
             startx=indexx
             starty=indexy
             current=env[indexx,indexy]
             goal=env[n-1,n-1]
             isgoal=start_node.is_goal(current,goal)   
            
a_star_search(env,0,0,goal)       