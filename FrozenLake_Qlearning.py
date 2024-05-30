import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygame

#渲染模式选择ansi,执行更快
env = gym.make('FrozenLake-v1',is_slippery = False,render_mode = "ansi")
q_table = np.zeros((env.observation_space.n,env.action_space.n))

episode = 20000

class QLearningAgent(object):
    def __init__(self) :
        self.action = [0,1,2,3]
        self.learning_rate = 0.8   #学习率
        self.discount_factor = 0.95 #折扣因子
        self.epsilon = 0.2         #贪心率
        self.q_table = q_table #frozenlake有16个state，4个action

    def learn(self,action,state,reward,next_state):
        current_q = self.q_table[state,action]
        #贝尔曼更新
        new_q = reward + self.discount_factor*np.max(self.q_table[next_state,:])
        #qlearning更新
        self.q_table[state,action] =self.q_table[state,action] + self.learning_rate*(new_q - current_q)

    #从Q表中选取动作
    #训练时采集动作
    def get_action(self,state):
        #np.random.rand():随机生成0-1间的数
        #epsilon-贪心策略随机选择动作
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action)
        else:
        #选择Q值最大的动作
            action = np.argmax(self.q_table[state,:])

            
        return action
    #测试时采集动作
    def get_action_test(self,state):
        #np.random.rand():随机生成0-1间的数
        #贪心策略采集动作
        action = np.argmax(self.q_table[state,:])
        return action

if __name__ == "__main__":
    #env = gym.make('FrozenLake-v1',is_slippery = False,render_mode = "ansi")
    #observation, info = env.reset()
    agent = QLearningAgent()
    #print(observation)
    x = []
    y = []
    R = 0
    for _ in range(episode):    
        state = env.reset()[0]
        
        while True:
            #采取动作
            action = agent.get_action(state)
            #print(action)
            #0~3:left,down,right,up
            observation, reward, terminated, truncated, info = env.step(action)
            
            next_state = observation
            agent.learn(action,state,reward,next_state)
            #修改q值，加快收敛
            #if next_state == state: #撞墙
                #q_table[state,action] = -1
            state = next_state
            x.append(_)
            R += reward
            y.append(R)
            if terminated or truncated:
                #if reward == 0: #掉进河里，q值为-1
                    #q_table[state,action] = -1
                #print(reward)
                #print(agent.q_table)
                observation, info = env.reset()
                break
    env.close()
    q_table_df = pd.DataFrame(q_table)    
    q_table_df.columns = ['left', 'down', 'right', 'up']
    print("Q_table after Training")
    print(q_table_df)  

    #plt.plot(x,y)
    #plt.title('cumulative_reward')
    #plt.xlabel('episode')
    #plt.ylabel('reward')

    # 显示图形
    #plt.show() 



    #根据学到的q表运行
    
    env = gym.make('FrozenLake-v1',is_slippery = False,render_mode = "human")
    agent_1 = QLearningAgent()
    state = env.reset()[0]
    for _ in range(10):
        while True:
                #采取动作
                action = agent_1.get_action_test(state)
                #print(action)
                #0~3:left,down,right,up
                observation, reward, terminated, truncated, info = env.step(action)
                next_state = observation
                state = next_state
                if terminated or truncated:
                    pygame.image.save(surface, 'saved_image.png')
                    observation, info = env.reset()
                    break
    env.close