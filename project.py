# Simulation
import numpy as np
import pandas as pd
from statistics import mean

n_T = 365
n_RDC = 37
n_notevalue = 2 # 0 = 50 note, 1 = 100 note
n_notetype = 3 # 0 = new, 1 = fit, 2 = unfit

T = range(n_T)
I = range(n_RDC)
J = range(n_notevalue)
B = range(n_notetype)

# Import demand forecasts
df = pd.read_csv(r"filepath\newDemand.csv")
df = df.iloc[:,1:]
newDemand = np.array(df.values)

df = pd.read_csv(r"filepath\fitDemand.csv")
df = df.iloc[:,1:]
fitDemand = np.array(df.values)

df = pd.read_csv(r"filepath\unfitDemand.csv")
df = df.iloc[:,1:]
unfitDemand = np.array(df.values)

# Convert demand from note value to amount of notes
demand = np.zeros((n_T, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
demand_list = [newDemand, fitDemand, unfitDemand]

for t in T:
    for i in I:
        for j in J:
            if j == 0:
                for b in B:
                    demand[t,i,j,b] = np.ceil(demand_list[b][t,i]*0.025/50)
            else:
                for b in B:
                    demand[t,i,j,b] = np.ceil(demand_list[b][t,i]*0.975/100)
    
# Shows which RDP does each RDC belongs to
df2 = pd.read_csv(r"filepath\Districts.csv")
df2 = df2.iloc[:,1:]
District = np.array(df2.values)

# Array containing all days in the year
T_all = np.array([t+1 for t in T])

# Weekdays (mondays) available for delivery
T_wd = np.array([10,17,24,38,45,52,59,66,73,80,87,101,108,115,129,136,143,150,
                 157,164,171,178,185,192,199,206,213,220,227,234,241,248,262,
                 269,283,290,297,304,311,318,325,332,339,346,353,360])
# Holidays (mondays) available for delivery
T_h = np.array([3,31,94,122,255,276])
# All days available for delivery
T_s = np.array([T_all[t] for t in T if (T_all[t] in T_wd) or (T_all[t] in T_h)])
n_Ts = len(T_s)
Ts = range(n_Ts)

# Indicator array for weekdays and holidays
T_di = np.zeros(len(T_s),dtype=np.int64)
for t in Ts:
    if T_s[t] in T_wd:
        T_di[t] = 0
    elif T_s[t] in T_h:
        T_di[t] = 1

# All holidays
i_h = np.array([1,2,3,31,32,33,34,35,36,37,93,94,95,120,121,122,123,124,154,	
                 155,156,253,254,255,274,275,276,277,278,279,280])

# All weekends
i_we = np.array([8,9,15,16,22,23,43,44,50,51,57,58,64,65,71,72,78,79,85,86,99,
                 100,106,107,113,128,134,135,141,142,148,149,162,163,169,170,
                 176,177,183,184,190,191,197,198,204,205,211,212,218,219,232,
                 233,239,240,246,247,260,261,267,268,288,289,295,296,302,303,
                 309,310,316,317,323,324,330,331,337,338,344,345,351,352,358,
                 359,365])

# All weekdays
wd_list = [T_all[t] for t in T if (T_all[t] not in i_we) and (T_all[t] not in i_h)]
i_wd = np.array(wd_list)

# Indentification of weekdays, weekends or holidays
T_i = np.zeros(n_T,dtype=np.int64)
for t in T:
    if T_all[t] in i_wd:
        T_i[t] = 0
    elif T_all[t] in i_we:
        T_i[t] = 1
    else:
        T_i[t] = 2

      
notevalue = np.array([50, 100])
notetype = np.array([1,2,3])

# Replenishment and callback costs, row 1: within district, row 2: cross, column 1: weekday, column 2: holiday
cost_rc = np.array([[195, 390], 
                    [254, 507]]) 
    
# Transfer costs
cost_tr = np.array([[150, 225, 300],
                    [195, 293, 390]]) #column 1: weekday, column 2: weekend, column 3: holiday

# Night courier
cost_nc = np.array([[300, 450, 600],
                    [390, 585, 780]])


c_max = 0.001 # overcap penalty multiplier
V = 1500000 # overcap limit for normal days
Vs = 5300000 # overcap limit for days in spring festival


# Initialise variables
X_r = np.zeros((n_T, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
X_c = np.zeros((n_T, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
R_t = np.zeros((n_T, n_RDC, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
X_n = np.zeros((n_T, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
Z = np.zeros((n_T, n_RDC),dtype=np.int64)

sim_c = [] # array to store costs of different replications


for i in range(1):
    waiting = False
    
    # initialise inventory
    Inv = np.zeros((n_T+1, n_RDC, n_notevalue, n_notetype),dtype=np.int64)
    
    for i in I:
        Inv[0, i, 0, 0] = np.ceil((20000/50)*0.05)
        Inv[0, i, 0, 1] = np.ceil((20000/50)*0.941)
        Inv[0, i, 0, 2] = np.ceil((20000/50)*0.009)
        Inv[0, i, 1, 0] = np.ceil((780000/100)*0.05)
        Inv[0, i, 1, 1] = np.ceil((780000/100)*0.941)
        Inv[0, i, 1, 2] = np.ceil((780000/100)*0.009)

    c = 0 # total cost
    c_r = 0 # total cost from replenishments and callbacks
    c_t = 0 # total cost from transfers
    c_n= 0 #  total cost from night couriers
    c_mm = 0 # total cost from overcap penalty
    N_w = 0 # Number of withdrawals
    N_s = 0 # Number of shortages
    
    s_t = 0.5 # proportion of upper bound target(s)
    
    # Set upper bound/lower bound targets for all notes
    S = np.zeros((n_notevalue, n_notetype+1),dtype=np.int64)
    S[0,0] = np.ceil((20400/50)*0.05)
    S[0,1] = np.ceil((20400/50)*0.941)
    S[0,2] = np.ceil((20400/50)*0.009)
    S[0,3] = V
    S[1,0] = np.ceil((788000/100)*0.05)
    S[1,1] = np.ceil((788000/100)*0.941)
    S[1,2] = np.ceil((788000/100)*0.009)
    S[1,3] = V
    s = S*s_t
    s = s.astype(np.int64)
    
    # Set upper bound/lower bound targets for all notes during spring festival
    SS = np.zeros((n_notevalue, n_notetype+1),dtype=np.int64)
    SS[0,0] = np.ceil((122000/50)*0.391)
    SS[0,1] = np.ceil((122000/50)*0.6)
    SS[0,2] = np.ceil((122000/50)*0.009)
    SS[0,3] = Vs
    SS[1,0] = np.ceil((4730000/100)*0.391)
    SS[1,1] = np.ceil((4730000/100)*0.6)
    SS[1,2] = np.ceil((4730000/100)*0.009)
    SS[1,3] = Vs
    ss = SS*s_t
    ss = ss.astype(np.int64)
    
    # Store targets in array
    upper = np.zeros((n_T, n_notevalue, n_notetype+1), dtype=np.int64)
    for t in T:
        if t in range(16,46):
            upper[t] = SS
        else:
            upper[t] = S

    lower = np.zeros((n_T, n_notevalue, n_notetype+1), dtype=np.int64)
    for t in T:
        if t in range(16,46):
            lower[t] = ss
        else:
            lower[t] = s

    for t in range(1,n_T):

        for i in I:
            Inv[t,i] = Inv[t-1,i] # update inventory to last period
            # Shipment arrival and departure if today is delivery day and update inventory 
            if t in T_s:
                waiting = False
                for j in J:
                    for b in B:
                        Inv[t,i,j,b] = Inv[t,i,j,b] + X_r[t,i,j,b] - X_c[t,i,j,b]
            if waiting == False:
                # Planning
                for d in Ts:
                    if T_s[d] > t: # obtain a future delivery date
                        
                        # check if inventory falls below trigger value
                        if np.sum(Inv[t,i,0,:2]) - np.sum(demand[t:T_s[d]+1,i,0,:2]) \
                        + np.sum(Inv[t,i,1,:2]) - np.sum(demand[t:T_s[d]+1,i,1,:2]) \
                        < np.sum(lower[T_s[d],:,:2]): 
                            for j in J:
                                for b in B:
                                    if b < 2:
                                        # Replenishments must bring inventory back to S
                                        X_r[T_s[d],i,j,b] = upper[T_s[d],j,b] + np.sum(demand[t:T_s[d]+1,i,j,b]) - Inv[t,i,j,b]
                                        c += cost_rc[District[i,0],T_di[d]] # update costs
                                        c_r += cost_rc[District[i,0],T_di[d]]
                        
                        # check if the total dollar value in a RDC exceeds max inventory limit
                        if (np.sum(Inv[t,i,0]) - np.sum(demand[t:T_s[d]+1,i,0]))*50 + (np.sum(Inv[t,i,1]) \
                            - np.sum(demand[t:T_s[d]+1,i,1]))*100 > upper[T_s[d],0,3]:
                            for j in J:
                                # callback all unfit notes
                                X_c[T_s[d],i,j,2] = Inv[t,i,j,2] - np.sum(demand[t:T_s[d]+1,i,j,2])
                                # callback all notes above the target
                                X_c[T_s[d],i,j,1] = max(0, Inv[t,i,j,1] - np.sum(demand[t:T_s[d]+1,i,j,1]) - upper[T_s[d],j,1])
                                X_c[T_s[d],i,j,0] = max(0, Inv[t,i,j,0] - np.sum(demand[t:T_s[d]+1,i,j,0]) - upper[T_s[d],j,0])
                                if X_c[T_s[d],i,j,1] > 0 or X_c[T_s[d],i,j,0] > 0 :
                                    c += cost_rc[District[i,0],T_di[d]]
                                    c_r += cost_rc[District[i,0],T_di[d]]
                                    
                        for j in J:
                            # check if unfit notes are over the target
                            if Inv[t,i,j,2] - np.sum(demand[t:T_s[d]+1,i,j,2]) > lower[T_s[d],j,2]:
                                # callback all unfit notes
                                X_c[T_s[d],i,j,2] = Inv[t,i,j,2] - np.sum(demand[t:T_s[d]+1,i,j,2])
                                c += cost_rc[District[i,0],T_di[d]]
                                c_r += cost_rc[District[i,0],T_di[d]]
                    
                        waiting = True
                        break
                    
            # Update inventory with demand
            Inv[t,i] -= demand[t,i]
            
            # Inter RDC transfers
            for j in J:
                for b in B:
                    if Inv[t,i,j,b] < 0: # check if inventory is below target
                        for l in I:
                            if Inv[t,l,j,b] > upper[t,j,b]: # check if inventory of other RDCs is above target
                                surplus = Inv[t,l,j,b] - upper[t,j,b]
                                deficit = upper[t,j,b] - Inv[t,i,j,b]
                                if surplus > deficit:
                                    R_t[t,l,i,j,b] = deficit
                                else:
                                    R_t[t,l,i,j,b] = surplus
                                Inv[t,i,j,b] += R_t[t,l,i,j,b] # transfer to and from
                                Inv[t,l,j,b] -= R_t[t,l,i,j,b]
                                c += cost_tr[District[l,i],T_i[t]]
                                c_t += cost_tr[District[l,i],T_i[t]]
                                N_w += 1
                                if Inv[t,i,j,b] >= 0: # end when positive inventory reached
                                    break
            
            # check if there is an overcap
            if np.sum(Inv[t,i,0])*50 + np.sum(Inv[t,i,1])*100 > upper[t,0,3]:
                c += c_max*(np.sum(Inv[t,i,0])*50 + np.sum(Inv[t,i,1])*100 - upper[t,0,3])
                c_mm += c_max*(np.sum(Inv[t,i,0])*50 + np.sum(Inv[t,i,1])*100 - upper[t,0,3])
            
            
            # Night courier used if inventory is still negative
            for j in J:
                for b in B:
                    if Inv[t,i,j,b] < 0:
                        X_n[t,i,j,b] = -Inv[t,i,j,b]
                        Inv[t,i,j,b] += X_n[t,i,j,b]
                        c += cost_nc[District[i,0],T_i[t]]
                        c_n += cost_nc[District[i,0],T_i[t]]
                        N_s += 1
                
    print("Total cost: ", c)
    print("Total withdrawal fulfiled: ", N_w)
    print("Total shortages occured: ", N_s)
    sim_c.append(c)
    
mean(sim_c)
