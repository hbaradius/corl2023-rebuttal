# corl2023-rebuttal

## Reviewer UVgt:
Thank you for acknowledging the strengths of our work. We address your comments below:

**C1:** ***"While the author considers operating within a latent space as a primary contribution of the paper (contribution 2), it is hard to claim so. Arguably, every neural network model, whether it is an expansion from a low to high dimensional space or a compression from high to low dimensional space, works with a latent space and the use of neural networks to model nonlinear relationships is not unique to this work."***

**A1:** We agree with the reviewer and will revise the text. Using latent representation is a common tactic in representation learning. Our original intention was to highlight that our approach performs -Bayesian filtering- over a learned latent representation. Still, we agree with the reviewer that this aspect is best explained in the main body and not made a contribution. Will revise!

**C2:** ***Additional information can be provided especially when $f_{\theta}$ takes in not only a sequence of previous states but also an optional action***

**A2:** A common transformer takes in a sequence of tokens as input. By default, we treat the past states as the sequence of inputs. When involving an additional action in the input, we simply treat the action as one additional token in the input sequence. The implementation details are included in Appendix A.2. 

**C3:** ***For the loss $L_S$, is such a loss possible? For example, in the multimodal manipulation task when $y^4$ is the force/torque sensor reading. Does the observation necessarily contain sufficient information to derive ground-truth state?***

**A3:** The latent representation for $y^4$ is learned via end-to-end training, since $y^4$ does not include the joint state of the robot and the object, we therefore did not place a condition on its latent representation.

**C4:** ***In the attention gain module, a matrix $\tilde{M}$ is introduced to retain only the diagonal elements of the attention map in support of the argument that "within each latent vector, every index is probabilistically independent, and index i of a latent state should only consider index i of each latent observation". Is there any evidence or prior works to support this claim?***

**A4:** We thank the reviewer for this excellent point. We conducted an empirical investigation of this hypothesis wherein we do not apply any masking on the $\tilde{M}$ matrix during training. As shown in table XXX, there is a clear and substantial deterioration in performance proposed method. Specifically, the MAE for joints increased from 5.24° to 16.25°. Moreover, the MAE for tracking the end effector also deteriorated from 3.04cm to 6.23cm. 

**C5:** ***The claim that "can be disabled based on different input sensor modalities, thus improving the model's resilience to missing modalities" also seem unconvincing as it is difficult to relate how the removal of makes the model resilient to missing modalities.***

**A5:** By disabling $\tilde{M}$, we selectively input some of the modalities to the model for training. This is similar to the training of Masked Language Models or Masked AutoEncoders: By training with missing inputs, the model tries to capture the joint distribution between each input part. Therefore, during inference time with missing modalities, the model has seen similar input distributions, thus it's able to perform more robustly.

[TODO add citation to Masked Autoencoders]

**C6:** ***Does $\alpha$-MDF only consider the RGB modality when compared with baselines dEFK and DPF in the estimation of the end-effector positions? It seems that only the RGB modality is used for dEKF since the DF baselines only take one modality as stated and the experiment (Table 1) does not consider sensor fusion techniques.***

**A6:** Yes, we make sure fair comparisons are carried out during experiments. When comparing with other differentiable filter baselines, i.e., dEKF and DPF, we train our proposed $\alpha$-MDF using only one modality, which is RGB. This provides a clean comparison.


## Reviewer zXri:
Thank you for your valuable feedback! Below, we address your comments and questions.

**C1:** ***"The proposed method seems heavily relying on vision sensors. From the ablation study, the IMU doesn’t seem to add much value to the system."***

**A1:** Yes, IMU contains certain noise when enough vision data is available especially for this case, the end-effector motion can be easily captured via depth and RGB. However, if we change the robot state into estimation the internal forces/acceleration of joints, then IMUs will be more reflective for such cases.

**C2:** ***"The organization of the paper can be improved. Currently, it’s a bit difficult to get a sense of the experimental setup without heavily referring to the appendix."***

**A2:** We thank the reviewer for pointing this out. We tried to put the information into the main paper as continuously as possible, but due to the page limit, many of the sections had to be moved to the appendix.

**C3:** ***"In section 3.1, lines 134-137, the paper states that the traditional Kalman gain is designed to handle a single observation space. I believe this is a bit inaccurate and an abuse of notation. How is the observation space defined here? The traditional Kalman filter does not restrict the measurements to be in the same measurement space (ex., We can simultaneously use velocity measurements and angular velocity measurements to correct the filter.) However, it is required that the function that maps from the state space to the measurement space be explicit. (As a result, some sensor measurements like images or deep-learned features are unable to be used in the formulation directly.) The proposed attention gain relaxes the need for an explicit measurement model."***

**A3:** We appreciate the reviewer for providing this question. We realize that concepts need to be defined with more details to allow the statements to be valid. 1. Traditional Kalman gain is used to trade-off between the state space (S )and the observation space (Y), which the K matrix only considers one observation space. Within the observation space, various types of observations can be defined, i.e., in KITTI, linear velocity and angular velocity used as observation.   2. In our paper, when we use the term "observation space", we mean each modality is encoded into one observation, where linear velocity and angular velocity can be encoded together into a latent observation. The attention gain is balanced over the state space (S) and multiple observations space (Y1, Y2, Y3).

**C4:** ***"Although mentioned in the limitation section, there’re no experiments demonstrating how well the proposed method works in cases of concept drifts. What happens if the surrounding environment changes drastically for the tensegrity robot experiments?"***

**A4:** The sensor encoders may fail to produce reliable latent representations from camera view if we applied changing background for the robot. There might be some vision models that can handle such cases, we provide the flexibility of the model to replace the sensor models into any advanced pretrained models.

**C5:** ***"I might have missed something, but for the manipulation task, why do joint angles need to be estimated as a state? Especially the joint angle measurements are used as an input to the network. The encoder measurements are usually fairly accurate already. I’m confused about why this is needed. In addition to the above point, the end-effector position can be computed through the forward kinematics function, although the kinematic modeling can sometimes be inaccurate. I wonder how the performance of the proposed method compares to pure forward kinematics."***

**A5:** In the appendix, we mentioned the joint angle are recorded with noise - this simulated a number of possible noise sources in the real world. It is important to note that even with rigid robot manipulators a variety of errors/noise sources can creep in [1,2] that cannot be resolved through accurate encoder measurements. This can be the result of: (a) calibration errors caused during the zeroing process, (b) errors in the hand-eye calibration, (c) differences in nominal and actual lengths of robot components, (d) changes to the camera pose due to physical forces exerted on the robot, (e) gearbox backlash and mechanical wear and tear, (f) the effects of temperature and payload weight on the mechanical structure of the robot. Often times small errors can compound throughout the kinematic chain and lead to positional errors in the centimeter range. In fact, addressing these issues during grasping and manipulation (in both rigid and soft robots) was the original question that eventually lead us to the writing of this paper. In our experiment, we therefore chose to understand how the filtering approach can deal with noisy measurement inputs.

[1] https://blog.robotiq.com/bid/72831/What-are-the-Sources-of-Robot-Inaccuracy
[2] Dantam et al. (2014), "Online Multi-Camera Registration for Bimanual Workspace Trajectories", Humanoids 2014, https://dyalab.mines.edu/papers/dantam2014multi.pdf

**C6:** ***"Since one of the main contributions of the proposed method is the attention gain, I’m interested to see how the proposed method works compared to a non-learning Kalman filter in cases where a traditional formulation is feasible. The Kalman gain is supposed to be the optimal gain under the Gaussian assumption and linear models. I wonder if the proposed attention gain can achieve similar optimal results when the problem is less complicated and well-constrained."***

**A6:** We agree with the reviewer that this will be an informative comparison. We added the suggested experiment, presenting a comparison of the results obtained from an (non-learning) Extended Kalman Filter (EKF) and α-MDF on the KITTI Visual Odometry task in low dimensions. As shown in Sec. B.1.2, our proposed method outperforms EKF by large margins. We argue that α-MDF has the additional capability of learning noise profiles during the training process, thereby eliminating the need for manual tuning, as required by the EKF.

**C7:** ***"Just out of curiosity, how are the sensors disabled for the experiment shown in Figure 5 (b)? It looks like the sensors were disabled on the fly during continuous inferencing. Was it done by setting the specific input matrix/vector to zero?"***

**A7:** Yes, it is correct. During inference, we only run the encoders of the existing modalities. For example, if there are only RGB and depth modalities, we only call the RGB encoder and depth encoder. Afterwards, we create only the corresponding $\tilde{M}$ matrix for RGB and depth. This results in a dynamic graph for the neural network, and thanks to modern deep learning frameworks such as TensorFlow2 and PyTorch, this is supported without much extra effort.

**C8:** ***"In Kitti odometry benchmark, it’s a common practice to estimate the 6D motion instead of x, y, and yaw, as the car might go up and down hills. I’m curious to learn how the proposed method works when the search space becomes larger."***

**A8:**


## Reviewer X85h:
We thank the reviewer for the comments and feedbacks. We also appreciate the reviewer for understanding and agreeing for the strengths of our proposed moethod!

Below, we address the comments.

**C1:** ***The paper has a few typographicals.***

**A1:** 
Thanks for pointing them out! We have fixed all the mentioned typos.

**C2:** ***"The paper lacks a comprehensive comparison in terms of computational complexity with current state-of-the-art models. The paper lacks details on the computational resources used for the experimental setup."***

**A2:** 

**C3:** ***"It would be beneficial for the paper if the authors could provide detailed sensitivity analysis and ablation studies to demonstrate the effectiveness of the different components of the proposed framework.The authors should provide more details and guidance on hyper-parameter tuning and its impact on the performance of the model."***

**A3:** We do agree with the reviewer on the importance of the sensitivity analysis. In Sec. B.2.2, we analyze the effects of three key factors on the performance of α-MDF. These factors are latent dimensions, the length of previous states, and the number of latent ensemble members. Our investigation focuses on understanding how these factors impact the overall performance of the α-MDF framework. We find that a larger latent dimension does not consistently yield better error metrics. Also, a longer history of states may not significantly contribute to estimating the current state. Therefore, the method is not sensitive to latent dimension and length of state history. However, As for the number of ensemble members, using a larger value for E does improve accuracy. It is also worth noting that increasing the number of ensemble members can result in a larger state space, which may introduce inefficiency.
