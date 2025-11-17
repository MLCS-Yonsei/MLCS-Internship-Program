import numpy as np

def get_state(state, action):
    
  action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  
  state[0]+=action_grid[action][0]
  state[1]+=action_grid[action][1]
  
  if state[0] < 0:
    state[0] = 0
  elif state[0] > 3:
    state[0] = 3
  
  if state[1] < 0:
    state[1] = 0
  elif state[1] > 3:
    state[1] = 3
  
  return state[0], state[1]

def policy_evaluation(grid_width, grid_height, action, policy, iter_num, reward=-1, dis=1):
    
  # table initialize
  post_value_table = np.zeros([grid_height, grid_width], dtype=float)
  
  # iteration
  if iter_num == 0:
    print('Iteration: {} \n{}\n'.format(iter_num, post_value_table))
    return post_value_table
  
  for iteration in range(iter_num):
    next_value_table = np.zeros([grid_height, grid_width], dtype=float)
    for i in range(grid_height):
      for j in range(grid_width):
        if i == j and ((i == 0) or (i == 3)):
          value_t = 0
        else :
          value_t = 0
          for act in action:
            i_, j_ = get_state([i,j], act)
            value = policy[i][j][act] * (reward + dis*post_value_table[i_][j_])
            value_t += value
        next_value_table[i][j] = round(value_t, 3)
    iteration += 1
    
    # print result
    if (iteration % 10) != iter_num: 
      # print result 
      if iteration > 100 :
        if (iteration % 20) == 0: 
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
      else :
        if (iteration % 10) == 0:
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
    else :
      print('Iteration: {} \n{}\n'.format(iteration, next_value_table ))
    
    post_value_table = next_value_table
      
  return next_value_table

grid_width = 4
grid_height = grid_width
action = [0, 1, 2, 3] # up, down, left, right
policy = np.empty([grid_height, grid_width, len(action)], dtype=float)
for i in range(grid_height):
  for j in range(grid_width):
    for k in range(len(action)):
      if i==j and ((i==0) or (i==3)):
        policy[i][j][k]=0.00
      else :
        policy[i][j][k]=0.25

value = policy_evaluation(grid_width, grid_height, action, policy, 100)

def policy_improvement(value, action, policy, reward = -1, grid_width = 4):
    
  grid_height = grid_width
  
  action_match = ['Up', 'Down', 'Left', 'Right']
  action_table = []
  
  # get Q-func.
  for i in range(grid_height):
    for j in range(grid_width):
      q_func_list=[]
      if i==j and ((i==0)or (i==3)):
        action_table.append('T')
      else:
        for k in range(len(action)):
          i_, j_ = get_state([i, j], k)
          q_func_list.append(value[i_][j_])
        max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)] 

        # update policy
        policy[i][j]= [0]*len(action) # initialize q-func_list
        for y in max_actions :
          policy[i][j][y] = (1 / len(max_actions))

        # get action
        idx = np.argmax(policy[i][j])
        action_table.append(action_match[idx])
  action_table=np.asarray(action_table).reshape((grid_height, grid_width))                
  
  print('Updated policy is :\n{}\n'.format(policy))
  print('at each state, chosen action is :\n{}'.format(action_table))
  
  return policy

updated_policy = policy_improvement(value, action, policy)