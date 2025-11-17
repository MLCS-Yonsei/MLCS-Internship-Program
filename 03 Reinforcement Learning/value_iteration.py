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
            value_t_list= []
            for act in action:
              i_, j_ = get_state([i,j], act)
              value = (reward + dis*post_value_table[i_][j_])
              value_t_list.append(value)
            next_value_table[i][j] = max(value_t_list)
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

value = policy_evaluation(grid_width, grid_height, action, policy, 1)
value = policy_evaluation(grid_width, grid_height, action, policy, 2)
value = policy_evaluation(grid_width, grid_height, action, policy, 3)
value = policy_evaluation(grid_width, grid_height, action, policy, 100)

