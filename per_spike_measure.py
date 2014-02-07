import multiprocessing as mp
import numpy as np
import lif_measures as lifm  # simulation script




if __name__=="__main__":
    print("Setting up ...")
    num_sims = 10
    num_inputs = 10+np.around(40*np.random.random(num_sims), 0)  # 10, 50
    input_frequencies = 10+np.around(90*np.random.random(num_sims), 0)  # 10, 100
    input_weights = np.around(5e-4*np.random.random(num_sims)+1)
    input_synchronies = np.around(np.random.random(num_sims), 2)  # 0, 1
    input_jitters = np.around(6e-3*np.random.random(num_sims), 4)  # 0, 6e-3
    params = zip(input_synchronies, input_jitters, num_inputs,
                 input_frequencies, input_weights)
    print("Simulations configured. Running ...")
    pool = mp.Pool()
    results_iter = pool.imap(lifm.lif_measures, params)


    # get measures through iterator
    # plot and save new figures every time a new data point arrives


