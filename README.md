# LQR-Trajectory-Tracker
Comparing Policy Search Methods for Optimal Quadcopter Control: Hooke-Jeeves vs Genetic Algorithms vs Cross Entropy
Comparing Policy Search Methods for Optimal Quadcopter Control: Hooke-Jeeves vs Genetic Algorithms vs Cross Entropy

Abstract: The goal of this paper is to compare three different policy search methods for the optimal control of quadcopter. Policy search methods are model free methods that search the space of policies without explicitly computing a value function. By utilizing a search of the policy space rather than the state space, the space is searched more efficiently due to the reduction in dimension. Three different policy search methods: Hooke Jeeves, Genetic algorithms, and Cross Entropy, were implemented for the optimal control of quadcopters. All three policy search methods performed admirably, creating stable paths that took the quadcopter around the obstacle and through wind conditions. Overall, it was found that the Cross-Entropy method provided the worst results in terms of path deviation, however it is more energy efficient compared to the other two methods and LQR. The other two methods, Hooke Jeeves and Genetic Algorithms, performed quite similarly when compared to LQR in terms of path deviation. Both methods were able to better adhere to the path. Additionally, both methods took similarly lower amounts of control energy. However, Hooke Jeeves took significantly less computational time when compared to genetic algorithms. Because of this, we recommend Hooke Jeeves as the most versatile policy search method.

A notebook featuring all code written for this project can be found here:
https://colab.research.google.com/drive/1cOaPJtXT6iJv68662KNsXDA_zF_35CZH?usp=sharing



Group Members: Roshan Jagani, Sunny Singh, Samantha Wu
AA 228: Decision Making under Uncertainty
December 8, 2023 
Table of Contents:

Problem Statement:	2

Existing Literature:	2

Approach:	3

Results:	6

Discussion and Conclusion:	10

Team Member Contributions:	11

References:	12

Problem Statement:
Quadcopter control is a good candidate for a decision-making problem due to the number of decisions a quadcopter must make, even after being given a set trajectory to reach its goal. These decisions include when to turn on each propeller and at what rate to spin each one.
The aim of this project is to compare different policy search methodologies, using the optimal control of quadcopters as the case study.  A simulated 2D quadcopter will be used to compare the performance of various policy search methodologies in regards to the 2D quadcopter’s adherence to a planned trajectory.  In this context the policy would be taken as the controller gain matrix.
Existing Literature:
A plethora of existing literature exists on the control of quadcopters. The most popular control strategy is Proportional Integral Derivative Control (PID). PID can be implemented to help stabilize the pitch, roll, and yaw movement of a quadcopter [1]. In “Modeling and Simulation of Quadcopter using PID Controller”, Praveen and Pillai create a 3D model of a quadcopter with motion in six directions: forward, backward, right, left, up, and down, and three rotations of axis: pitch, roll, and yaw. In the controller model, Praveen and Pillai used pitch, roll, or yaw as the reference signal and output to the quadcopter. It seems the gains were manually tuned for the PID controller.
Another strategy for the control of quadcopter is reinforcement learning. Reinforcement learning utilizes a model of well-trained neural networks to map state to actuator commands [2]. In “Control of a Quadrotor with Reinforcement Learning”, Hwangbo et al. utilize reinforcement learning for control of a quadrotor through a unique approach that only utilizes a model in simulation. The methodology is technically model free since the policy is not trained using a model. Instead, the policy is trained with a value and policy network. The exploration strategy utilizes trajectories separated into three categories: initial, junction, and branch. The value function is trained using monte-carlo samples obtained from on-policy trajectories. The policy is then optimized using a natural gradient descent. In simulated testing this strategy out performed traditional learning algorithms and showcased a very stable system.
An alternative to reinforcement learning is a model free approach of policy search methodologies. In “Augmented Random Search for Quadcopter Control: An alternative to Reinforcement Learning”, a model-free approach called Augmented Random Search (ARS) is explored as an alternative to model-based reinforcement learning strategies. Tiwari et al. were able to get a conservative but stable controller for quadcopter utilizing a deterministic policy. This model-free approach utilized a random search of the policy space which performed admirably when compared to an agent trained with the model based reinforcement learning technique: Deep Deterministic policy gradient (DDPG). The ARS agent was able to learn the goal faster and more accurately than the DDPG agent. This shows model-free policy search to be a promising alternative to model based reinforcement learning.
Approach:
Building from a previous homework assignment in AA274A, a simulated 2D quadcopter model will be utilized to compare three different policy search methodologies: Hooke Jeeves, Genetic Algorithms, and the cross-entropy method. From the previous literature search, it was determined that model-free approaches perform admirably when compared to model-based approaches. Therefore, all of the implemented methodologies will be model-free in an attempt to search for the best policy search method.

In the virtual experiment, we first set up a class for simulating the dynamics of a planar quadrotor (drone) and for visualizing its trajectory. Then we implemented a direct method to compute a nominal trajectory for the quadrotor to move from a starting position to an end position while avoiding an obstacle. For context, the nominal trajectory assumes that the environment has no disturbances and that the plan can be followed perfectly. We then simulated what would occur if a quadrotor followed this nominal trajectory in the presence of a wind disturbance. Then we used the Decision-Making techniques to implement an “optimal” LQR algorithm that allows the quadrotor to correct its course as it tries to track the nominal trajectory in the presence of the wind disturbance.

For the decision-making challenge, we used a multi-objective cost function. In particular, we had 2 objectives: the first of which was to minimize the deviation of the drone’s trajectory from the optimal trajectory if there were no uncertainties (i.e. no wind disturbances). We chose to use Euclidean distance as our metric for deviation because we believe it to be more intuitive than Manhattan distance for this model of the planar quadcopter. The second objective was to minimize the control torques applied to the rotors in order to also optimize for energy efficiency. We chose to compute the cost of energy as the mean of the entire control sequence in order to avoid our objective function being overshadowed by an extreme control input at one or two time steps. Furthermore, these two objectives were equally weighted in the aggregate objective function. The aggregate objective function takes in a gain matrix and returns the total cost as the reward:
R=(xopen-xclosed)2+(yopen-yclosed)2  + 1ni=1nu(i) 

We were then able to use this objective function to compare multiple different policy search methods and explore which one would improve our gain matrix the most. We did this by modeling our experiment as a sequential problem. This allowed us to use Policy Search methods to solve the sequential problem. However, even though all the methods we compare below are Policy search methods, they come from different families of optimization problems. For example, Hooke-Jeeves is a direct method, while the genetic algorithm is a population method, and lastly Cross-Entropy is a Stochastic method.


a. Hooke Jeeves
The Hooke Jeeves method is a local search algorithm that assumes the policy is parameterized by an n-dimensional vector θ. The algorithm then takes a step in each coordinate direction from θ. If no improvements are found in that step, the step size is reduced until improvements are found. Once an improvement is found then it moves to that best point. This continues until the step size drops below some preselected threshold[4]. Algorithm 10.2 from “Algorithms for Decision Making” was implemented in python, with the aforementioned hyperparameters of step size, reduction factor, and improvement threshold being adjusted heuristically until the algorithm produced dependable convergence to a suitable solution. The algorithm requires a starting point, which was taken as the first gain matrix produced by the LQR controller. The hyperparameters are summarized in the table below.

Table 1: Hyperparameters for Hooke-Jeeves implementation
Algorithm
Step Size
Reduction Factor
Improvement Threshold
Hooke-Jeeves
2
0.5
0.01





b. Genetic Algorithm
The genetic algorithm policy search method aims to improve upon the Hooke Jeeves method by minimizing the possibility of getting stuck in a local optimum. Genetic algorithms are inspired from nature’s natural selection. The genetic algorithm iteratively updates the policy parameterizations based on the previous iterations best samples[4]. An initial population of randomly and uniformly sampled policies is evaluated for its best performing members. This best performing 50% of members are selected as parents, and are paired up randomly to produce a new generation of offspring policies. There are many potential options for implementing this process, called crossover, but we selected crossover with interpolation, in which the offspring of two parents is a linear interpolation between the parameters of each parent. Crossover is performed between random pairs of parents until a new generation of policies is created with the desired population size. Finally, a mutation step is performed to randomly alter a select few parameters of the new generation. This step introduces stochasticity and reduces the likelihood of the algorithm getting stuck in a local optimum for lack of exploration of the design space. This cycle is repeated for a desired number of generations, and the best performing policy from the final generation is taken as the optimized gain matrix. The algorithm hyperparameters are described in the table below.

Table 2: Hyperparameters for Genetic Algorithm implementation
Algorithm
Pop. Size
Crossover Rate
Mutation Rate
Generations
Genetic Algorithm
50
0.8
0.01
100



c. Cross Entropy
The final methodology implemented was the cross entropy method. This policy search method updates a search distribution over the parameterized space of policies at each iteration[4]. In this stochastic method, at each iteration a set of m samples are sampled from a proposal distribution, which is then updated for the next iteration based on the best performing samples. The chosen proposal distribution was a normal distribution, with the mean μ and covariance matrix ∑ being updated according to the maximum likelihood estimate:
(k+1)=1melitei=1melitex(i)	(k+1)=1melitei=1melite(x(i)-(k+1))(x(i)-(k+1))T
Where melite is the chosen number of elite samples taken at each iteration. The parameters m and melite were chosen heuristically as values that produced reliable convergence to a consistent global optimum, and are tabulated below in Table 3. We opted to initialize the algorithm with a larger number of samples to provide better coverage of the design space.

Table 3: Hyperparameters for Cross Entroup implementation
Algorithm
M
M_elite
Initial Samples
Iterations
Cross Entropy
100
20
0.01
100


Results:
The results of these policy search techniques are presented below in the form of simulated quadcopter trajectories as they fly around an obstacle in a randomized wind field. The blue path represents the nominal goal trajectory, while the orange trajectory is the actual path taken by the quadcopter during the simulation.
All three policy search methods resulted in functional gain matrices that took the quadcopter around the obstacle while it remained stable and controllable. However each algorithm resulted in a different path deviation error, energy cost, and
computational time, as shown below in Table 4.

Figure 1: Orange quadcopter path is Open-Loop Trajectory. Blue path is goal path


Figure 2: Orange quadcopter path is Closed-Loop Trajectory without any policy search. Blue path is goal path


Figure 3: Orange quadcopter path is Hooke Jeeves Trajectory. Blue Path is goal path


Figure 4:Orange quadcopter path is Genetic Algorithm Trajectory. Blue Path is goal path


Figure 5: Orange quadcopter path is Cross Entropy Trajectory. Blue Path is goal path

Table 4: Summary of path deviation, control energy, and computation time for Hooke-Jeeves, Genetic Algorithm, Cross-Entropy, Open Loop, and LQR algorithms
Algorithm
Path Deviation
Control Energy
Computation Time (s) 
Hooke-Jeeves
0.3166
13.236
3.566 
Genetic Algorithm
0.3164
13.221
66.162
Cross-Entropy
1.184
12.115
154.973
Open Loop
1.5358
N/A
N/A
LQR w/o Policy Search
0.37122
13.780
0.360


Discussion and Conclusion:
Following the implementation of three different policy search methodologies: Hooke Jeeves, Genetic Algorithm, and Cross Entropy, it was found that even though all three algorithms are model-free and Policy Search methods, they still had visually different results. Most notably, the Cross-entropy approach performed the worst in terms of Path Deviation. However, on the other hand, when we consider the energy consumption from the controls generated by this approach, it was slightly more efficient than the other methods. 

The other two Policy Search methods, Hooke-Jeeves and the Genetic Algorithm, both tracked the trajectory more accurately as they had less path deviation than LQR. These two methods also had less energy cost than vanilla LQR but the drawback is that they took longer computation time to generate these more optimal trajectories. We would recommend Hooke-Jeeves as the optimal approach that balances the tradeoffs between deviation and energy cost along with computation time.

One of the factors that we would like to explore in future work is further tuning the objective function. More specifically, we can modify the objective function to weight path deviation more than energy consumption to get more precise trajectory tracking results.

Team Member Contributions:
Roshan Jagani: 
Roshan’s contribution to this project was evenly splitting the Approach, Analysis, and results sections, including implementing the optimization algorithms in Python

Sunny Singh: 
Sunny’s contribution to this project was evenly splitting the Approach, Analysis, and results sections, including implementing the objective function in Python

Samantha Wu: 
Samantha’s contribution to this project included concentrating on exploring previous literature written on our chosen topic (control of quadcopters), crafting the abstract, problem statement, approach, and conclusion for the paper, as well as general group organization. 
References:
[1] Viswanadhapalli, Praveen, and Anju S. Pillai. Modeling and Simulation of Quadcopter Using PID Controller. International Science Press, 2016, pp. 7151–258.

[2] Hwangbo, Jemin, and Sa, Inkyu et al. “Control of a Quadrotor With Reinforcement Learning”. In: IEEE Robotics and Automation Letters 2.4 (Oct. 2017), pp. 2096–2103. ISSN: 2377-3774. DOI: 10. 1109/lra.2017.2720851.7.

[3] Tiwari, Ashutosh Kumar, and Nadimpalli, Sandeep Varma “Augmented Random Search for Quadcopter Control: An Alternative to Reinforcement Learning.” International Journal of Information Technology and Computer Science, vol. 11, no. 11, Nov. 2019, pp. 24–33. DOI.org (Crossref), https://doi.org/10.5815/ijitcs.2019.11.03.

[4] Kochenderfer, Mykel J., et al. Algorithms for Decision Making. The MIT Press, 2022.







