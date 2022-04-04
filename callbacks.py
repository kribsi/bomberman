import copy
import os
import pickle
import random
import numpy as np

MY_ACTIONS = ["coin", "bomb opponent", "bomb crate", "find action"]

def setup(self):
    self.total_coins=9  #CHANGE TO 9 LATER!
    self.subdivision_size=13
    self.overall_discovered_coin_positions=[]
    self.overall_collected_coins=0
    self.set_bomb=-1
    self.action_history = ""
    self.width, self.height =17,17
    self.some_position=(1,1)
    self.features=()
    self.prev_features = ()
    self.action=''
    self.prev_action=''
    self.step =0
    self.current=(None,None)
    self.coins_visible_in_game=0
    self.coins_left_in_game=self.total_coins
    self.map = np.zeros(shape=(self.width,self.height), dtype=str)
    self.possible_steps = np.zeros((self.width,self.height, 5), dtype=bool)
    #the positions for actions:
    self.safe_coin_available = False
    self.safe_coin_position = ()
    self.best_proportioned_coin = ()
    self.opponent_dest_pos = ()
    self.crate_dest_pos = ()
    self.action_position = ()
    self.A,self.B,self.C=[], [], np.zeros(shape=(self.width,self.height, 5))

    self.my_action=""
    self.my_prev_action = ""
    self.state = None
    self.prev_state = None
    self.crates=np.zeros(shape=(self.width, self.height), dtype=int)
    self.coins=np.zeros(shape=(self.width, self.height), dtype=int)

    if self.train or not os.path.isfile('my-saved-model.pt'):
        self.logger.info('Setting up model from scratch.')
        weights = np.random.rand(len(MY_ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info('Loading model from saved state.')
        with open('my-saved-model.pt', 'rb') as file:
            self.model = pickle.load(file) 

def act(self, game_state: dict) -> str:
    determine_size(self, game_state)
    make_crates(game_state, self)
    make_coins ( game_state, self)
    set_positions_of_others(self, game_state)
    update_set_bomb(self)
    update_overall_discovered_coins(self, game_state)
    self.prev_features = self.features
    self.my_prev_action=self.my_action
    self.current=game_state['self'][3]
    self.possible_steps, self.map = map_out(self,game_state, self.current) 
    self.A,self.B,self.C =explore_component(self.current, self)
    set_some_position(self, self.current)
    analyse(self,game_state)
    self.state= calculate_state_number(self.features)
    T= decide_action(self) 
    return T
    

def update_overall_discovered_coins(self, game_state):  
    self.coins_visible_in_game=len(game_state['coins'])
    if (game_state['step']<self.step): 
        self.overall_discovered_coin_positions=[]
        self.overall_collected_coins=0
    for coin in game_state['coins'] :
        if coin not in self.overall_discovered_coin_positions : 
            self.overall_discovered_coin_positions.append(coin)
    self.overall_collected_coins = len(self.overall_discovered_coin_positions)-self.coins_visible_in_game
    self.step = game_state['step']
    self.coins_left_in_game=self.total_coins-self.overall_collected_coins

def update_set_bomb(self):
    if (self.set_bomb!=-1): 
        if (self.set_bomb<4) : self.set_bomb=self.set_bomb+1
        else : self.set_bomb=-1

def set_features (self,game_state, closest_opponent_to_coin, coin_list) :
    my_pos=self.current
    subdivision= closest_n_fields(self.subdivision_size, my_pos, self) 
    self.opponent_dest_pos, opponent_val =destruction_potential(self,subdivision, game_state, opponent_destruction_number) 
    self.crate_dest_pos, crate_val=destruction_potential(self,subdivision, game_state, crate_destruction_number)  
    self.coin_pos, coin_val = coin_finding(self, closest_opponent_to_coin, coin_list)
    events=number_events_in_subdivision(subdivision, self) 
    self.features= almost_features_to_features (self.coins_left_in_game, coin_val, opponent_val, crate_val, events) 

def coin_finding(self, closest_opponent_to_coin, coin_list):
    extracted_coin_distances =np.zeros(shape=len(coin_list))
    for i in range(len(coin_list)):
        extracted_coin_distances[i]=coin_list[i][1]
    proportions =extracted_coin_distances/closest_opponent_to_coin
    if(len(coin_list)!=0):
        best_proportion = np.amin(proportions)
        best_index= (np.where(proportions==best_proportion))[0][0]
        best_proportioned_coin, dist_to_best_proportioned_coin = coin_list[best_index]
    else : 
        dist_to_best_proportioned_coin=9999
        best_proportioned_coin = np.where(closest_n_fields(13, self.current, self)==1)[:][0]    
    return best_proportioned_coin,dist_to_best_proportioned_coin

def almost_features_to_features (coins_left_in_game, coin_val, opponent_val, crate_val, number_events_in_subdivision):
    
    #feature_coins_left
    if (coins_left_in_game==0): feature_coins_left=0
    if (coins_left_in_game==1): feature_coins_left=1
    if (coins_left_in_game in range(2,6)): feature_coins_left=2
    if (coins_left_in_game>5): feature_coins_left=3

    #feature_dist_best_proportioned_coin        #SHOULDNT THIS RATHER BE THE PROPORTION??? NO BECAUSE OTHERS MIGHT BE TOO DUMB TO FIND THE COINS. LETS LEAVE IT LIKE THAT.
    if (coin_val<6) : feature_dist_best_proportioned_coin=0
    if (coin_val in range(6,10)) : feature_dist_best_proportioned_coin=1
    if (coin_val in range(10, 30)) : feature_dist_best_proportioned_coin=2
    if (coin_val>=30) : feature_dist_best_proportioned_coin=3

    #feature_opponent_destruction_potential
    if (opponent_val==0) : feature_opponent_destruction_potential=0
    if (opponent_val==1) : feature_opponent_destruction_potential=2
    if (opponent_val>=2) : feature_opponent_destruction_potential=3

    #feature_crate_destruction_potential
    if(crate_val==0) : feature_crate_destruction_potential=0
    if(crate_val in range(1,3)) : feature_crate_destruction_potential=1
    if(crate_val in range(3,5)) : feature_crate_destruction_potential=2
    if(crate_val>=5) : feature_crate_destruction_potential=3
    
    #feature_events_in_subdivision
    if (number_events_in_subdivision==0): feature_events_in_subdivision=0
    if (number_events_in_subdivision==1): feature_events_in_subdivision=1
    if (number_events_in_subdivision==2): feature_events_in_subdivision=2
    if (number_events_in_subdivision>=3): feature_events_in_subdivision=3

    return (feature_coins_left, 
            feature_dist_best_proportioned_coin, 
            feature_opponent_destruction_potential, 
            feature_crate_destruction_potential, 
            feature_events_in_subdivision)

def learnt_alg (self) :   
    
    #return "coin"
    #return np.random.choice(MY_ACTIONS, p=[.25,.25,.25,.25])

    random_prob = 0.5
    if self.train and random.random() < random_prob:
        self.logger.debug('Choosing action purely at random.')
        return np.random.choice(MY_ACTIONS, p=[.25,.25,.25,.25])

    self.logger.debug('Querying model for action.')
    return propose_action(self)

def propose_action (self) : 
    with open("my-saved-model.pt", "rb") as file:
            QTABLE, V = pickle.load(file)
    list=[]
    for action in MY_ACTIONS :
        list.append(QTABLE[(self.state, action)])
    if (list[0]==list[1]==list[2]==list[3]) : return np.random.choice(MY_ACTIONS, p=[.25,.25,.25,.25])
    if list.index(max(list))==0 : return "coin"
    if list.index(max(list))==1 : return "bomb opponent"
    if list.index(max(list))==2 : return "bomb crate"
    if list.index(max(list))==3 : return "find action"
    return "model doesnt work"

def decide_action(self) :
    self.my_action = learnt_alg(self) 

    if self.safe_coin_available :
        #ignore learned part 
        goal=self.safe_coin_position
        self.action = find_first_step_of_shortest_path(self,self.current,goal)[0] 
        return self.action  

    # execute learned part    
    if (self.my_action=="coin"): 
        if (self.set_bomb>-1): 
            goal= self.some_position 
        else: 
            goal = self.best_proportioned_coin 
    if (self.my_action=="bomb opponent"): 
        if (self.set_bomb>-1): 
            goal= self.some_position 
        elif (self.opponent_dest_pos==self.current) :
            self.action= "BOMB"
            start_bomb_timer(self)
            return self.action
        else:
            goal= self.opponent_dest_pos
    if (self.my_action=="bomb crate"):
        if (self.set_bomb>-1): 
            goal = self.some_position
        elif (self.crate_dest_pos==self.current) :
            self.action= "BOMB"
            start_bomb_timer(self)
            return self.action
        else :
            goal = self.crate_dest_pos
    if (self.my_action=="find action"): 
        goal = self.action_position
    self.action = find_first_step_of_shortest_path(self,self.current,goal)[0] 
    return self.action 

def start_bomb_timer(self):
    self.set_bomb=0

def analyse (self,game_state) :
    self.safe_coin_available = False
    self.safe_coin_position = ()
    positions_of_others = array_to_position_list (self.others, self)
    coin_list = order_goals_by_distance(self,self.current, game_state['coins']) 
    closest_opponent_to_coin=99999
    for coin in coin_list : 
        if (self.safe_coin_available==False):
            x,y =coin[0]
            my_dist = coin[1]
            ordered_opponents =order_goals_by_distance(self,coin[0], positions_of_others)
            if (ordered_opponents!=[]):
                closest_opponent_to_coin=ordered_opponents[0][1]
            if (my_dist<=closest_opponent_to_coin) : 
                self.safe_coin_available = True
                self.safe_coin_position = (x,y)
    big_subdiv= closest_n_fields(50, self.current, self)
    if ((self.coins_left_in_game!=0) &(self.crates!=0).any()) : 
        action_position = destruction_potential(self, big_subdiv, game_state, crate_destruction_number)[0]
    elif (game_state['others']!=[]): 
        action_position = destruction_potential(self, big_subdiv, game_state, opponent_destruction_number)[0]
    else: action_position=self.some_position
    self.action_position = action_position
    set_features(self,game_state, closest_opponent_to_coin, coin_list)
    if (self.best_proportioned_coin ==()) : self.best_proportioned_coin = self.some_position
    return

def array_to_position_list(array, self):
    list = []
    for i in range(self.width):
        for j in range(self.height):
            d=array[i,j]
            if (d!=0) : list.append((i,j))
    return list

def number_events_in_subdivision(subdivision, self) :
    events = (subdivision+1)*(np.add(self.others,self.crates,self.coins))
    return events.sum()

def set_positions_of_others(self,game_state) : 
    list=game_state['others']
    others = np.zeros(shape=(self.width, self.height), dtype=int)
    for other in list :
        others[other[3]]=1
    self.others = others
    return

def closest_n_fields(subdivision_size, my_pos, self) :    
    def explore(i, Q, A, subdivision, d, a, b):
        if ((a,b) not in A) :
            Q.append((a,b))
            A.append((a,b))
            subdivision[a,b]=d+1
            i=i+1
        return i
    i=1
    Q=[my_pos]
    A=Q.copy()
    subdivision=np.zeros(shape=(self.width, self.height))
    subdivision.fill(-1)
    subdivision[my_pos]=0
    for d in range(17^2):
        P=Q.copy()
        Q=[]
        if (i<subdivision_size) :
            for position in P:
                r, l, u, dow, stay = self.possible_steps[position]
                if r : 
                    if (i<subdivision_size) :
                        a,b = right(position)
                        i = explore(i, Q, A, subdivision, d, a, b)
                    else: break
                if l : 
                    if (i<subdivision_size) :
                        a,b = left(position)
                        i = explore(i, Q, A, subdivision, d, a, b)
                    else: break
                if u : 
                    if (i<subdivision_size) :
                        a,b = up(position)
                        i = explore(i, Q, A, subdivision, d, a, b)
                    else: break
                if dow : 
                    if (i<subdivision_size) :
                        a,b = down(position)
                        i = explore(i, Q, A, subdivision, d, a, b)
                    else: break
        else: break
    return (subdivision)

def crate_destruction_number (pos, self) :
    destroyed_crates = self.crates*affected_positions(pos, self, self.map)
    return (destroyed_crates!=0).sum()

def opponent_destruction_number (pos, self) :  
    a,b, component=explore_component(pos, self)
    component=10-component
    destroyed_opponents = self.others*affected_positions(pos, self, self.map)*component
    return (destroyed_opponents!=0).sum()  

def destruction_potential(self, subdivision, game_state, destruction_number_function) :
    destruction = np.zeros(shape=(self.width, self.height), dtype = int)
    for i in range(self.width):
        for j in range(self.height) :
            if (subdivision[i,j]!=-1):
                pos=(i,j)
                destruction[i,j] = destruction_number_function(pos, self)
    best=(None, None)
    value=None
    while(best==(None, None)):
        value = np.amax(destruction) 
        best=(np.where(destruction==value)[0][0], np.where(destruction==value)[1][0])
        if value==0 :
            return (self.some_position, 0)
        if (bomb_check(self, best, game_state)==False) : 
            destruction[best]=0
            best=(None, None)
    return (best, value)


def bomb_check(self, bomb_position, game_state) :
    f=(bomb_position, 4)
    hypothetical_game_state = copy.deepcopy(game_state)
    hypothetical_game_state['bombs'].append(f)
    hypothetical_possible_steps, hypothetical_map=map_out(self,hypothetical_game_state, bomb_position)
    possible_to_escape=False 
    if(hypothetical_possible_steps[bomb_position].sum()>0) :
        possible_to_escape = True
    return possible_to_escape

def map_out(self,game_state, current) :
    map = make_map(self, game_state)
    possible_steps = make_possible_steps(self, map, current)    
    danger_reach = find_danger_reach(game_state, self, map)
    possible_steps = disqualify_dangerous_steps(self,current, 6, danger_reach, possible_steps)
    return (possible_steps, map)

def make_map (self, game_state) :
    map = np.zeros(shape=(self.width,self.height), dtype=str)
    for i in range(self.width) :
        for j in range(self.height): 
            if (game_state['field'][i,j] == 1) : map[i,j] = 'c'
            elif (game_state['field'][i,j] == -1) : map[i,j] = 'w'
            else : 
                map[i,j] = 'f'
            for bomb in game_state['bombs'] :
                if (bomb[0]==(i,j)) : map[i,j] = str(bomb[1])
            for other in game_state['others'] :
                if (other[3] == (i,j)) : map[i,j] = 'o'
    return map    

def find_danger_reach(game_state, self, map):
        danger_reach = np.zeros(shape=(self.width, self.height, 7))
        for t in range(5):
            for bomb in game_state['bombs'] : 
                if (bomb[1]==t) : 
                    affected = affected_positions(bomb[0], self, map)
                    danger_reach[affected!=0, (t+1):(t+3)] = 1
        just_exploded_locations = (game_state['explosion_map']==1)
        danger_reach[just_exploded_locations,0:2] = 1
        return danger_reach

def affected_positions(bomb_position, self, map) :        
    current_field=bomb_position
    result = np.zeros(shape=(self.width, self.height), dtype = int)
    result[current_field]=1
    i=0
    while ((i<3) and (map[current_field]!='w')) :
        i=i+1
        result[right(current_field)]=1
        current_field=right(current_field)
    current_field=bomb_position
    i=0
    while ((i<3) and (map[current_field]!='w')) :
        i=i+1
        result[left(current_field)]=1
        current_field=left(current_field)
    current_field=bomb_position
    i=0
    while ((i<3) and (map[current_field]!='w')) :
        i=i+1
        result[up(current_field)]=1
        current_field=up(current_field)
    current_field=bomb_position
    i=0
    while ((i<3) and (map[current_field]!='w')) :
        i=i+1
        result[down(current_field)]=1 
        current_field=down(current_field)  
    return (result)  

class Node ():
    def __init__(me,x,y, possible_steps): 
        me.x=x
        me.y=y
        me.deadly=False
        me.number_of_safe_children = Node.number_safe_children(x,y,possible_steps)
        me.dist=None
        me.came_from=me
        me.initial_step=""
    def number_safe_children(x,y,possible_steps) :
        return possible_steps[x,y,:].sum()

def disqualify_dangerous_steps (self,current, depth: int, danger_reach, possible_steps): 
    def explore(danger_reach, possible_steps, Q, A, d, p, a, b):
        new=Node(a,b, possible_steps)
        new.dist, new.came_from= d+1, p
        if (danger_reach[a,b,d+1]!=0): 
            new.deadly = True
        Q.append(new)
        A.append(new) 
    x,y= current
    s=Node(x,y, possible_steps)
    Q=[]
    Q.append(s)
    A=Q.copy()
    for d in range(depth):
        P=Q.copy()
        Q=[]
        for p in P:
            r, l, u, dow, stay = possible_steps[p.x,p.y,:]
            position=(p.x,p.y)
            if r : 
                a,b = right(position)
                explore(danger_reach, possible_steps, Q, A, d, p, a, b)
            if l : 
                a,b=left(position) 
                explore(danger_reach, possible_steps, Q, A, d, p, a, b)                   
            if u : 
                a,b=up(position)  
                explore(danger_reach, possible_steps, Q, A, d, p, a, b)
            if dow : 
                a,b=down(position) 
                explore(danger_reach, possible_steps, Q, A, d, p, a, b)
            if stay :
                a,b=position 
                explore(danger_reach, possible_steps, Q, A, d, p, a, b)
    for i in range(depth-1):
        d=depth-i
        for a in A:
            if ((a.dist==d) & (a.deadly==True)) : 
                (a.came_from).number_of_safe_children=(a.came_from).number_of_safe_children-1
        for a in A:
            if (a.dist==d-1) :
                    if (a.number_of_safe_children==0) : 
                        a.deadly = True 
    for a in A:
        if ((a.dist ==1) & (a.deadly==True)) : 
            if (right((x,y))==(a.x,a.y)):
                possible_steps[x,y,0]=False
            if (left((x,y))==(a.x,a.y)):
                possible_steps[x,y,1]=False
            if (up((x,y))==(a.x,a.y)):
                possible_steps[x,y,2]=False
            if (down((x,y))==(a.x,a.y)):
                possible_steps[x,y,3]=False
            if (x,y)==(a.x,a.y):
                possible_steps[x,y,4]=False
    return possible_steps 



def order_goals_by_distance(self,my_pos,positions_of_goals) :
    def explore_pos(positions_of_goals, Q, A, dist, d, pos):
        if (pos not in A) :
            Q.append(pos)
            A.append(pos)
            if (pos in positions_of_goals): 
                dist.append((pos,d+1))
    Q=[(my_pos)]
    A=Q.copy()
    dist=[]
    for d in range(17^2):
        P=Q.copy()
        Q=[]
        for position in P:
            r, l, u, dow, stay = self.possible_steps[position]
            pos = right(position)
            if r : explore_pos(positions_of_goals, Q, A, dist, d, pos)
            pos=left(position)
            if l : explore_pos(positions_of_goals, Q, A, dist, d, pos)
            pos=up(position)   
            if u : explore_pos(positions_of_goals, Q, A, dist, d, pos)
            pos=down(position)
            if dow: explore_pos(positions_of_goals, Q, A, dist, d, pos)
    return (dist)



def explore_component (pos, self) : 
    x,y= pos
    s=Node(x,y, self.possible_steps)
    Q=[]
    Q.append(s)
    B=Q.copy()
    A=[(x,y)]
    C=np.zeros(shape=(self.width, self.height), dtype= int)
    C.fill(-1)
    C[x,y]=0
    for d in range(17^2+3):
        P=Q.copy()
        Q=[]
        for p in P:
            r, l, u, dow, stay = self.possible_steps[p.x,p.y,:]
            position=(p.x,p.y)
            if r : 
                a,b = right(position)
                if ((a,b) not in A) :
                    new=Node(a,b, self.possible_steps)
                    new.dist, new.came_from= d+1, p
                    if (d==0) : new.initial_step = "RIGHT"
                    else : new.initial_step = (new.came_from).initial_step 
                    Q.append(new)
                    A.append((a,b))
                    B.append(new)
                    C[a,b]=d+1
            if l : 
                a,b=left(position)
                if ((a,b) not in A) : 
                    new=Node(a,b, self.possible_steps)
                    new.dist, new.came_from = d+1, p 
                    if (d==0) : new.initial_step = "LEFT"
                    else : new.initial_step = (new.came_from).initial_step
                    Q.append(new)
                    A.append((a,b))
                    B.append(new)
                    C[a,b]=d+1
            if u : 
                a,b=up(position)  
                if ((a,b) not in A) :
                    new=Node(a,b, self.possible_steps)
                    new.dist, new.came_from = d+1, p
                    if (d==0) : new.initial_step = "UP"
                    else : new.initial_step = new.came_from.initial_step
                    Q.append(new)
                    A.append((a,b))
                    B.append(new)
                    C[a,b]=d+1
            if dow : 
                a,b=down(position) 
                if ((a,b) not in A) :
                    new=Node(a,b, self.possible_steps)
                    new.dist, new.came_from = d+1, p
                    if (d==0) : new.initial_step = "DOWN"
                    else : new.initial_step = new.came_from.initial_step
                    Q.append(new)
                    A.append((a,b))
                    B.append(new)
                    C[a,b]=d+1
    return (A, B, C)

def find_first_step_of_shortest_path (self, start, goal): 
        x,y=start
        if (goal not in self.A) : 
            #print(goal, "impossible, not in component")
            return ("impossible, not in component", None)
        else: 
            if goal == (x,y):
                steps_here=self.possible_steps[x,y,:]
                if steps_here[4]==True : return ("STAY", 0)
                if steps_here[0]==True : return ("RIGHT", 0)
                if steps_here[1]==True : return ("LEFT", 0)
                if steps_here[2]==True : return ("UP", 0)
                if steps_here[3]==True : return ("DOWN", 0)
                return ("impossible, because component just our position (all steps disqualified)", 0)
            for node in self.B:
                if ((node.x, node.y)==goal) : 
                    return (node.initial_step, node.dist)
    
def make_possible_steps (self, map, current) :
    def walkable(pos, map) : 
        i,j=pos
        if (map[i,j] not in ['c', 'w', 'o', '4', '3', '2', '1', '0']) : return True
        return False
    possible_steps = np.zeros((self.width,self.height, 5), dtype=bool)
    for i in range(self.width) :
        for j in range(self.height):
            pos=(i,j)
            possible_steps[i,j,:]=False, False, False, False, True #necessary!
            if ((walkable(pos, map)==True) | ((pos==current) & (map[current] in {'4', '3', '2', '1'}))) : 
                if walkable(right(pos), map) : 
                    possible_steps[i,j,0]=True
                if walkable(left(pos), map) : 
                    possible_steps[i,j,1]=True
                if walkable(up(pos), map) : 
                    possible_steps[i,j,2]=True
                if walkable(down(pos), map) : 
                    possible_steps[i,j,3]=True
    return possible_steps

def determine_size(self,game_state):
        self.width, self.height= game_state['field'].shape

def up(position) :
        x,y=position
        return (x,y-1)
def down(position) : 
        x,y=position
        return (x,y+1)
def left(position) : 
        x,y=position
        return (x-1,y)
def right(position) : 
        x,y=position
        return (x+1,y)

def make_crates (game_state, self) :
    crates=np.zeros(shape=(self.width, self.height), dtype=int)
    crates[game_state['field']==1]=1
    self.crates = crates

def make_coins(game_state, self) :
    coins = np.zeros(shape=(self.width, self.height), dtype=int)
    for coin in game_state['coins'] :
        coins[coin]=1
    self.coins=coins

def set_some_position (self,my_pos) :
    i=1
    goal=self.A[len(self.A)-i]
    while(find_first_step_of_shortest_path(self,my_pos,goal)=="impossible"):
        i=i+1
        goal = self.A[len(self.A)-i]
    self.some_position=goal

def calculate_state_number(array) :
        i,j,k,l,m= array
        return i*10000+j*1000+k*100+l*10+m