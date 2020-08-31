import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import pandas as pd

env = gym.make('Taxi-v3').env
frames = []

tries = 0

penalties, rewards = 0, 0

done = False
while not done:
	action = env.action_space.sample()
	state, reward, done, info = env.step(action)

	if reward == -10:
		penalties += 1

	frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

	tries += 1

print('Time Taken : {}'.format(tries))
print('Number of penalties: {}'.format(penalties))

def print_frames(frames):
	for i, frame in enumerate(frames):
		clear_output(wait=True)
		print(frame['frame'])
		print(f"Timestep: {i + 1}")
		print(f"State: {frame['state']}")
		print(f"Action: {frame['action']}")
		print(f"Reward: {frame['reward']}")
		sleep(.3)

qtable = np.zeros([500,6])

learning_rate = 0.5
discount_rate = 0.8
epsilon = 0.2


for i in range(1, 10000):
	state = env.reset()

	tries, penalties, rewards = 0, 0, 0
	finished = False

	while not finished:
		if np.random.uniform(0, 1) < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(qtable[state])

		nexts, reward, finished, info = env.step(action)

		old = qtable[state, action]
		nxt = np.max(qtable[nexts])
		newval = (1 - learning_rate)*old + learning_rate * (reward + discount_rate * nxt)		
		qtable[state, action] = newval

		if reward == -10:
			penalties += 1

		state = nexts
		tries += 1

	if i % 100 == 0:
		clear_output(wait=True)
		print(f"Episode: {i}")

test_tries, test_penalties = 0, 0
number_of_tests = 100

df = pd.DataFrame(qtable)
df.to_csv('qtable', sep=',')

video = []
state = env.reset()
runs, penalties, rewards = 0, 0, 0

finished = False
while not finished:
	action = np.argmax(qtable[state])
	state, reward, finished, info = env.step(action)

	if reward == -10:
		penalties += 1

	video.append({
	    'frame': env.render(mode='ansi'),
	    'state': state,
	    'action': action,
	    'reward': reward
	    }
	   )
	runs += 1

	test_penalties += penalties
	test_tries += runs

print_frames(video)
print(f"Penalties Incurred {penalties}")
print(f'Took {runs} steps to complete the dropoff(s)')
