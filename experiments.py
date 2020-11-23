from hiive.mdptoolbox import mdp
from mdptoolbox import example
import gym
import matplotlib.pyplot as plt
import numpy as np



label_dict = {0: 'Left',
              1: 'Down',
              2: 'Right',
              3: 'Up',
              4: 'Hole'}

def convert_gym(env):
    P_gym = env.P
    new_P = np.zeros([env.nA, env.nS, env.nS])
    new_R = np.zeros([env.nS, env.nA])
    for s_0 in P_gym:
        this_state = P_gym[s_0]
        for a in this_state:
            this_sa = this_state[a]
            for sas in this_sa:
                p = sas[0]
                s_1 = sas[1]
                r = sas[2]
                new_P[a, s_0, s_1] += p
                new_R[s_0, a] += p*r
    return new_P, new_R


def plot_values(V, size, env):
    # reshape value function
    V_sq = np.reshape(V, (size,size))
    mask = np.array(env.desc) == b'H'
    V_sq[mask] = 4

    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j,i),label in np.ndenumerate(V_sq):
        ax.text(i, j, label_dict[label], ha='center', va='center', fontsize=6)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    plt.show()


small = gym.make('FrozenLake-v0').unwrapped
P_small, R_small = convert_gym(small)

# region VI

# avg V, n_iter, time
ep_vals = [.1, .0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    for gamma in gamma_vals:
        vi = mdp.ValueIteration(P_small, R_small, gamma=gamma, epsilon=epsilon)
        stats = vi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)


for i in range(len(ep_vals)):
    plt.plot(gamma_vals, big_vs[i], label="Epsilon=" + str(ep_vals[i]))
plt.title("Average V's at varying gamma and epsilon")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FL_small_vi_Vs.png")


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
f.suptitle("Convergence Time")
for i in range(len(ep_vals)):
    ax1.plot(gamma_vals, big_n[i], label="Epsilon=" + str(ep_vals[i]))
ax1.set_title("Iterations to convergence at varying gamma and epsilon")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")


for i in range(len(ep_vals)):
    ax2.plot(gamma_vals, big_t[i], label="Epsilon=" + str(ep_vals[i]))
ax2.set_title("Time to convergence at varying gamma and epsilon")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_small_vi_CC.png")



# Get optimal policy
vi = mdp.ValueIteration(P_small, R_small, gamma=gamma_vals[-1], epsilon=ep_vals[-1])
vi.run()


# Plot optimal policy
policy = vi.policy
plot_values(policy, 4, small)
plt.savefig("FL_small_vi_pol.png")

# Plot V over time
avg_vs = []
for stat in vi.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_small_vi_conv.png")

# endregion

# region PI


# avg V, n_iter, time
ep_vals = [.0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    for gamma in gamma_vals:
        pi = mdp.PolicyIteration(P_small, R_small, gamma=gamma)
        stats = pi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)



plt.plot(gamma_vals, big_vs[0], label="Epsilon=" + str(ep_vals[0]))
plt.title("Average V's at varying gamma")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FL_small_pi_Vs.png")


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
f.suptitle("Convergence Time")
ax1.plot(gamma_vals, big_n[0], label="Epsilon=" + str(ep_vals[0]))
ax1.set_title("Iterations to convergence at varying gamma")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")


ax2.plot(gamma_vals, big_t[0], label="Epsilon=" + str(ep_vals[0]))
ax2.set_title("Time to convergence at varying gamma")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_small_pi_CC.png")



# Get optimal policy
pi = mdp.PolicyIteration(P_small, R_small, gamma=gamma_vals[-1])
pi.run()


# Plot optimal policy
policy = pi.policy
plot_values(policy, 4, small)
plt.savefig("FL_small_pi_pol.png")

# Plot V over time
avg_vs = []
for stat in pi.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_small_pi_conv.png")

# endregion

# region Q Learning


# avg V, n_iter, time
alpha_vals = [.1, .3, .5, .7, .9]
epslion_vals = [.2, .4, .6, .8]

big_vs = []
big_n = []
big_t = []
for epsilon in epslion_vals:
    avg_vs = []
    n_iters = []
    times = []
    for alpha in alpha_vals:
        q = mdp.QLearning(P_small, R_small, gamma=.9999, alpha=alpha, alpha_decay=1, epsilon=epsilon, epsilon_decay=.99)
        stats = q.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)


for i in range(len(epslion_vals)):
    plt.plot(alpha_vals, big_vs[i], label="Epsilon=" + str(epslion_vals[i]))
plt.title("Average V's at varying epsilon and alpha")
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("Average V")
plt.savefig("FL_small_q_Vs.png")


f, ax2 = plt.subplots(1, 1, figsize=(12,6))
f.suptitle("Convergence Time")


for i in range(len(epslion_vals)):
    ax2.plot(alpha_vals, big_t[i], label="Epsilon=" + str(epslion_vals[i]))
ax2.set_title("Time to convergence at varying epsilon and alpha")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_small_q_CC.png")



# Get optimal policy
q = mdp.QLearning(P_small, R_small, n_iter=100000, gamma=.9999, alpha=.5, alpha_decay=.99999, epsilon=.5, epsilon_decay=.99)
q.run()


# Plot optimal policy
policy = q.policy
plot_values(policy, 4, small)
plt.savefig("FL_small_q_pol.png")

# Plot V over time
avg_vs = []
for stat in q.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_small_q_conv.png")


# endregion


Lake20x20 = ['SHFFFFHFFHFFFFFFFFFF',
             'FFFHFFFFFHFFFFFHFFFF',
             'FFFFFFHFFFFFFFFFFFFF',
             'FFHFFFFFFFFFFFFFFFFF',
             'FFFFFFFFHFFFFFFFFFFF',
             'HFFFFFFFFFFFFFHFFFFF',
             'FFFFFFFFFHFFFFFFFFHF',
             'FFFFFFFFFFFFHFFFFFFF',
             'FFFHFFFFFFFFFFFFHFFF',
             'FFFFHFFFFFFFFFFFFFFF',
             'FFFFHFFHFFFFHFFHFFFH',
             'FFFFFFFFFFHFFFFFFFFF',
             'FFFFFFFFFFFFFFFFFFFF',
             'FFFFFFFFFFFFFFFFFFFF',
             'FFFFFFFHFFFFFFFFFFFF',
             'FFFFFFHFFFFFFFFFFFFF',
             'HFFFFFFFFFFFFFFHFFFF',
             'FHFFFFFFFFFFFFFHFHFH',
             'FFFFFFHFFFFFFFFFFFFF',
             'FFFHFFFFHFFFFFFFFHFG']



large = gym.make('FrozenLake-v0', desc=Lake20x20).unwrapped
P_large, R_large = convert_gym(large)



# region VI

# avg V, n_iter, time
ep_vals = [.1, .0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    for gamma in gamma_vals:
        vi = mdp.ValueIteration(P_large, R_large, gamma=gamma, epsilon=epsilon)
        stats = vi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)


for i in range(len(ep_vals)):
    plt.plot(gamma_vals, big_vs[i], label="Epsilon=" + str(ep_vals[i]))
plt.title("Average V's at varying gamma and epsilon")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FL_large_vi_Vs.png")


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
f.suptitle("Convergence Time")
for i in range(len(ep_vals)):
    ax1.plot(gamma_vals, big_n[i], label="Epsilon=" + str(ep_vals[i]))
ax1.set_title("Iterations to convergence at varying gamma and epsilon")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")


for i in range(len(ep_vals)):
    ax2.plot(gamma_vals, big_t[i], label="Epsilon=" + str(ep_vals[i]))
ax2.set_title("Time to convergence at varying gamma and epsilon")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_large_vi_CC.png")



# Get optimal policy
vi = mdp.ValueIteration(P_large, R_large, gamma=gamma_vals[-1], epsilon=ep_vals[-1])
vi.run()


# Plot optimal policy
policy = vi.policy
plot_values(policy, 20, large)
plt.savefig("FL_large_vi_pol.png")

# Plot V over time
avg_vs = []
for stat in vi.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_large_vi_conv.png")

# endregion

# region PI


# avg V, n_iter, time
ep_vals = [.0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    for gamma in gamma_vals:
        pi = mdp.PolicyIteration(P_large, R_large, gamma=gamma)
        stats = pi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)



plt.plot(gamma_vals, big_vs[0], label="Epsilon=" + str(ep_vals[0]))
plt.title("Average V's at varying gamma")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FL_large_pi_Vs.png")


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
f.suptitle("Convergence Time")
ax1.plot(gamma_vals, big_n[0], label="Epsilon=" + str(ep_vals[0]))
ax1.set_title("Iterations to convergence at varying gamma")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")


ax2.plot(gamma_vals, big_t[0], label="Epsilon=" + str(ep_vals[0]))
ax2.set_title("Time to convergence at varying gamma")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_large_pi_CC.png")



# Get optimal policy
pi = mdp.PolicyIteration(P_large, R_large, gamma=gamma_vals[-1])
pi.run()


# Plot optimal policy
policy = pi.policy
plot_values(policy, 20, large)
plt.savefig("FL_large_pi_pol.png")

# Plot V over time
avg_vs = []
for stat in pi.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_large_pi_conv.png")

# endregion

# region Q Learning


# avg V, n_iter, time
alpha_vals = [.3, .5, .7, .9, .95]
epslion_vals = [.2, .4, .6, .8]

big_vs = []
big_n = []
big_t = []
for epsilon in epslion_vals:
    avg_vs = []
    n_iters = []
    times = []
    for alpha in alpha_vals:
        q = mdp.QLearning(P_large, R_large, gamma=.9999, n_iter=20000, alpha=alpha, alpha_decay=.9999, epsilon=epsilon, epsilon_decay=.999)
        stats = q.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)


for i in range(len(epslion_vals)):
    plt.plot(alpha_vals, big_vs[i], label="Epsilon=" + str(epslion_vals[i]))
plt.title("Average V's at varying epsilon and alpha")
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("Average V")
plt.savefig("FL_large_q_Vs.png")


f, ax2 = plt.subplots(1, 1, figsize=(12,6))
f.suptitle("Convergence Time")


for i in range(len(epslion_vals)):
    ax2.plot(alpha_vals, big_t[i], label="Epsilon=" + str(epslion_vals[i]))
ax2.set_title("Time to convergence at varying epsilon and alpha")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FL_large_q_CC.png")



# Get optimal policy
q = mdp.QLearning(P_large, R_large, n_iter=2000000, gamma=.9999, alpha=.9, alpha_decay=.9999, epsilon=.2, epsilon_decay=.999)
q.run()


# Plot optimal policy
policy = q.policy
plot_values(policy, 20, large)
plt.savefig("FL_large_q_pol.png")

# Plot V over time
avg_vs = []
for stat in q.run_stats:
    avg_v = stat['Mean V']
    avg_vs.append(avg_v)
plt.plot(avg_vs)
plt.title("Average V Value Over Time")
plt.xlabel("Iteration Number")
plt.ylabel("Average V")
plt.savefig("FL_large_q_conv.png")


# endregion




# Forest Management



P, R = example.forest(S=10, r1=2, r2=1, p=.1)



# region VI

# avg V, n_iter, time
ep_vals = [.1, .0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
big_p = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    pps = []
    for gamma in gamma_vals:
        vi = mdp.ValueIteration(P, R, gamma=gamma, epsilon=epsilon)
        stats = vi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)
        pps.append(vi.policy)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)
    big_p.append(pps)


for i in range(len(ep_vals)):
    plt.plot(gamma_vals, big_vs[i], label="Epsilon=" + str(ep_vals[i]))
plt.title("Average V's at varying gamma and epsilon")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FM_vi_Vs.png")


f, ax1 = plt.subplots(1, 1, figsize=(12,6))
f.suptitle("Convergence Time")
for i in range(len(ep_vals)):
    ax1.plot(gamma_vals, big_n[i], label="Epsilon=" + str(ep_vals[i]))
ax1.set_title("Iterations to convergence at varying gamma and epsilon")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")


f.savefig("FM_vi_CC.png")



# Plot policies
policy = big_p[1]
for i in range(len(gamma_vals)-1):
    plt.plot(policy[i], label="Gamma=" + str(gamma_vals[i]))
plt.legend()
plt.title("Cut (1) or not (0) for different gamma values and states")
plt.xlabel("State")
plt.ylabel("Cut?")
plt.savefig("FM_vi_pol.png")

# endregion

# region PI


# avg V, n_iter, time
ep_vals = [.0001]
gamma_vals = [.2, .5, .8, .95, .999]

big_vs = []
big_n = []
big_t = []
big_p = []
for epsilon in ep_vals:
    avg_vs = []
    n_iters = []
    times = []
    pps = []
    for gamma in gamma_vals:
        pi = mdp.PolicyIteration(P, R, gamma=gamma)
        stats = pi.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)
        pps.append(pi.policy)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)
    big_p.append(pps)



plt.plot(gamma_vals, big_vs[0], label="Epsilon=" + str(ep_vals[0]))
plt.title("Average V's at varying gamma")
plt.legend()
plt.xlabel("Gamma")
plt.ylabel("Average V")
plt.savefig("FM_pi_Vs.png")


f, ax1 = plt.subplots(1, 1, figsize=(12,6))
f.suptitle("Convergence Time")
ax1.plot(gamma_vals, big_n[0], label="Epsilon=" + str(ep_vals[0]))
ax1.set_title("Iterations to convergence at varying gamma")
ax1.legend()
ax1.set_xlabel("Gamma")
ax1.set_ylabel("N iterations")

f.savefig("FM_pi_CC.png")



# Plot policies
policy = big_p[0]
for i in range(len(gamma_vals)-1):
    plt.plot(policy[i], label="Gamma=" + str(gamma_vals[i]))
plt.legend()
plt.title("Cut (1) or not (0) for different gamma values and states")
plt.xlabel("State")
plt.ylabel("Cut?")
plt.savefig("FM_pi_pol.png")

# endregion

# region Q Learning


# avg V, n_iter, time
alpha_vals = [.1, .3, .5, .7, .9]
epslion_vals = [.2, .4, .6, .8]

big_vs = []
big_n = []
big_t = []
for epsilon in epslion_vals:
    avg_vs = []
    n_iters = []
    times = []
    pps = []
    for alpha in alpha_vals:
        q = mdp.QLearning(P, R, gamma=.9, alpha=alpha, alpha_decay=1, epsilon=epsilon, epsilon_decay=.99)
        stats = q.run()

        avg_v = stats[-1]['Mean V']
        n_iter = len(stats)
        time = stats[-1]['Time']

        avg_vs.append(avg_v)
        n_iters.append(n_iter)
        times.append(time)

    big_vs.append(avg_vs)
    big_n.append(n_iters)
    big_t.append(times)


for i in range(len(epslion_vals)):
    plt.plot(alpha_vals, big_vs[i], label="Epsilon=" + str(epslion_vals[i]))
plt.title("Average V's at varying epsilon and alpha")
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("Average V")
plt.savefig("FM_q_Vs.png")


f, ax2 = plt.subplots(1, 1, figsize=(12,6))
f.suptitle("Convergence Time")


for i in range(len(epslion_vals)):
    ax2.plot(alpha_vals, big_t[i], label="Epsilon=" + str(epslion_vals[i]))
ax2.set_title("Time to convergence at varying epsilon and alpha")
ax2.legend()
ax2.set_xlabel("Gamma")
ax2.set_ylabel("Time")
f.savefig("FM_q_CC.png")

# endregion
