import numpy as np 
import matplotlib.pyplot as plt
import FTRL
import sys
from FTRL import player
# Police game social welfare plot idea.
# plot num players initially calling vs. social welfare of the eventual equilibrium.

'''
2p police game:

[(1,1),(1,2)]
[(2,1),(0,0)]
Nashes: .5 .5, 0 1, 1 0

'''
'''
My understanding of the police game: Each player's strategy is some 0<=p<=1. payoff -1 for calling, 2 for someone calling.
Social welfare: 2n*Pr(someone calls)-E[number of callers]
OPT: 2n-1
calling probabilities p1,p2,...,pn
SW: 2n*prod(1-pk) - sum(pk)

'''
num_players = 10
def random_bias(num_players):
	biases = []
	for k in range(num_players):
		ran = np.random.rand()
		biases.append(np.array([ran,1-ran]))
	return biases
def create_biases(num_players,num_init_callers):
	assert num_init_callers <= num_players
	biases = []
	for k in range(num_players):
		u = np.array([0,0])
		if k < num_init_callers:
			u[1]=1.
		else:
			u[0] = 1.
			#u = np.array([0.5,0.5])
		biases.append(u)
	return biases
def social_welfare(action_list):
	numplayers = len(action_list)
	sw = 0.
	sw += 2*numplayers
	for _ in range(numplayers):
		sw *= action_list[_][1]
	sw = 2*numplayers - sw
	for _ in range(numplayers):
		sw -= action_list[_][0]
	return sw
def opt_social_welfare(num_players):
	return 2*num_players - 1

def run_nash(num_players, biases,epochs):
	#N_players = 4
	def payoff_func(action,other_actions): #[0=not call,1=call]
		pr_no_call = 1.
		for ac in other_actions:
			pr_no_call *= ac[0]

		pr_call = 1.-pr_no_call
		util = action[1] + action[0]*(pr_no_call*0 + pr_call*2)
		return util

	# players = []
	# for p in range(num_players):
	# 	new_player = player(identity=p,n_actions=2,payoff_func=payoff_func,regularizer=FTRL.quad_regularizer,epochs=epochs)
	# 	players.append(new_player)
	# for i in range(epochs):
	# 	if i % 20 == 0 and i > 0:
	# 		for p in players:
	# 			print 'most recent', p.get_most_recent()
	# 			print 'average', p.get_avg_action()
	# 	FTRL.update_FTRL(players,tick=i,box=False,mixed=True)

	# for p in players:
	# 	print 'most recent', p.get_most_recent()
	# 	print 'average', p.get_avg_action()
	# exit()


	players = []
	for p in range(num_players):
		new_player = player(identity=p,n_actions=2,payoff_func=payoff_func,regularizer=FTRL.quad_regularizer,epochs=epochs,regularizer_bias=biases[p])
		players.append(new_player)

	for i in range(epochs):	
		#if i % 20 == 0:
		#	print 'iteration',  i
		#	for p in players:
		#		print 'most recent', p.get_most_recent()
		if i % 60 == 0 and i > 0:
			print i
			for p in players:
				pass
				# print 'most recent', p.get_most_recent()
		FTRL.update_FTRL(players, tick=i, box=False,mixed=True)
		#FTRL.MWU(players,epochs=epochs,tick=i,mixed=True)

	final_actions = []
	for p in players:
		final_actions.append(p.get_most_recent())
	# print final_actions
	return final_actions

k_array = []
sw_array = []	
non_normalized = []
def projection(actionvector):
	return np.transpose(np.transpose(actionvector)[0])
for k in range(num_players+1):
	# print 'number of people biased to call', k
	# biases = create_bias(num_players,k)
	# biases = random_bias(num_players)
	biases = [[0.2,0.8],[0.2,.8],[0.2,0.8],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]
	print projection(biases)
	resulting_action_list = run_nash(num_players, biases,epochs=100)
	print projection(resulting_action_list)
	quit()
	k_array.append(k)
	non_normalized.append(social_welfare(resulting_action_list))
	sw_array.append(social_welfare(resulting_action_list)*1./opt_social_welfare(num_players))
plt.plot(k_array,sw_array)
plt.show()
plt.plot(k_array, non_normalized)
plt.show()
