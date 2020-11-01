# Awesome Interaction-aware Behavior and Trajectory Prediction
![Version](https://img.shields.io/badge/Version-1.1-ff69b4.svg) ![LastUpdated](https://img.shields.io/badge/LastUpdated-2020.11-lightgrey.svg)![Topic](https://img.shields.io/badge/Topic-trajectory--prediction-yellow.svg?logo=github)

This is a checklist of state-of-the-art research materials (datasets, blogs, papers and public codes) related to trajectory prediction. Wish it could be helpful for both academia and industry. (Still updating)

**Maintainers**: [**Jiachen Li**](https://jiachenli94.github.io), Hengbo Ma, Jinning Li (University of California, Berkeley)

**Emails**: {jiachen_li, hengbo_ma, jinning_li}@berkeley.edu

Please feel free to pull request to add new resources or send emails to us for questions, discussion and collaborations.

**Note**: [**Here**](https://github.com/jiachenli94/Awesome-Decision-Making-Reinforcement-Learning) is also a collection of materials for reinforcement learning, decision making and motion planning.



Please consider citing our work if you found this repo useful:

```
@inproceedings{li2020evolvegraph,
  title={EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning},
  author={Li, Jiachen and Yang, Fan and Tomizuka, Masayoshi and Choi, Chiho},
  booktitle={2020 Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@inproceedings{li2019conditional,
  title={Conditional Generative Neural System for Probabilistic Trajectory Prediction},
  author={Li, Jiachen and Ma, Hengbo and Tomizuka, Masayoshi},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6150--6156},
  year={2019},
  organization={IEEE}
}
```

### Table of Contents

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
- [**Datasets**](#datasets)
	- [Vehicles and Traffic](#vehicles-and-traffic)
	- [Pedestrians](#pedestrians)
	- [Sport Players](#sport-players)
- [**Literature and Codes**](#literature-and-codes)
	- [Survey Papers](#survey-papers)
	- [Physics Systems with Interaction](#physics-systems-with-interaction)
	- [Intelligent Vehicles & Traffic](#intelligent-vehicles-traffic)
	- [Pedestrians](#pedestrians-1)
	- [Mobile Robots](#mobile-robots)
	- [Sport Players](#sport-players)
	- [Benchmark and Evaluation Metrics](#benchmark-and-evaluation-metrics)
	- [Others](#others)
	<!-- /TOC -->

## **Datasets**
### Vehicles and Traffic

|                           Dataset                            |            Agents            |         Scenarios         |        Sensors         |
| :----------------------------------------------------------: | :--------------------------: | :-----------------------: | :--------------------: |
|      [INTERACTION](http://www.interaction-dataset.com/)      | Vehicles / cyclists/ people  | Roundabout / intersection |     Camera     |
|        [KITTI](http://www.cvlibs.net/datasets/kitti/)        | Vehicles / cyclists/ people  |   Highway / rural areas   |     Camera / LiDAR     |
|           [HighD](https://www.highd-dataset.com/)            |           Vehicles           |          Highway          |         Camera         |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) |           Vehicles           |          Highway          |         Camera         |
| [Cyclists](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Tsinghua-Daimler_Cyclist_Detec/tsinghua-daimler_cyclist_detec.html) |           Cyclists           |           Urban           |         Camera         |
|            [nuScenes](https://www.nuscenes.org/)             |           Vehicles           |           Urban           | Camera / LiDAR / RADAR |
|          [BDD100k](https://bdd-data.berkeley.edu/)           | Vehicles / cyclists / people |      Highway / urban      |         Camera         |
| [Apolloscapes](http://apolloscape.auto/?source=post_page---------------------------) | Vehicles / cyclists / people |           Urban           |         Camera         |
| [Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets) |           Vehicles           |           Urban           |         Camera         |
|      [Cityscapes](https://www.cityscapes-dataset.com/)       |       Vehicles/ people       |           Urban           |         Camera         |
| [Stanford Drone](http://cvgl.stanford.edu/projects/uav_data/) | Vehicles / cyclists/ people  |           Urban           |         Camera         |
|           [Argoverse](https://www.argoverse.org/)            |      Vehicles / people       |           Urban           |     Camera / LiDAR     |
| [TRAF](https://gamma.umd.edu/researchdirections/autonomousdriving/trafdataset)            |      Vehicles/buses/cyclists/bikes / people/animals       |           Urban           |     Camera      |
|[Lyft Level 5](https://level5.lyft.com/dataset/)               | Vehicles/cyclists/people     | Urban                     | Camera/ LiDAR   |

### Pedestrians

|                           Dataset                            |           Agents            |       Scenarios       |    Sensors     |
| :----------------------------------------------------------: | :-------------------------: | :-------------------: | :------------: |
| [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) |           People            |    Zara / students    |     Camera     |
|       [ETH](http://www.vision.ee.ethz.ch/en/datasets/)       |           People            |         Urban         |     Camera     |
|              [VIRAT](http://www.viratdata.org/)              |      People / vehicles      |         Urban         |     Camera     |
|        [KITTI](http://www.cvlibs.net/datasets/kitti/)        | Vehicles / cyclists/ people | Highway / rural areas | Camera / LiDAR |
|     [ATC](https://irc.atr.jp/crest2010_HRI/ATC_dataset/)     |           People            |    Shopping center    |  Range sensor  |
| [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html) |           People            |  From moving vehicle  |     Camera     |
| [Central Station](http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html) |           People            |    Inside station     |     Camera     |
| [Town Center](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets) |           People            |     Urban street      |     Camera     |
| [Edinburgh](http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/) |           People            |         Urban         |     Camera     |
|   [Cityscapes](https://www.cityscapes-dataset.com/login/)    |      Vehicles/ people       |         Urban         |     Camera     |
|           [Argoverse](https://www.argoverse.org/)            |      Vehicles / people      |         Urban         | Camera / LiDAR |
| [Stanford Drone](http://cvgl.stanford.edu/projects/uav_data/) | Vehicles / cyclists/ people |         Urban         |     Camera     |
|           [TrajNet](http://trajnet.stanford.edu/)            |           People            |         Urban         |     Camera     |
|           [PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/)            |           People            |         Urban         |     Camera     |
|           [ForkingPaths](https://next.cs.cmu.edu/multiverse/index.html)            |           People            |         Urban / Simulation         |     Camera     |
|           [TrajNet++](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge)            |           People            |         Urban         |     Camera     |
### Sport Players

|                           Dataset                            | Agents |     Scenarios     | Sensors |
| :----------------------------------------------------------: | :----: | :---------------: | :-----: |
|     [Football](https://datahub.io/collections/football)      | People |  Football field   | Camera  |
| [NBA SportVU](https://github.com/linouk23/NBA-Player-Movements) | People |  Basketball Hall  | Camera  |
|      [NFL](https://github.com/a-vhadgar/Big-Data-Bowl)       | People | American Football | Camera  |

## **Literature and Codes**

### Survey Papers

- Modeling and Prediction of Human Driver Behavior: A Survey, 2020. [[paper](https://arxiv.org/abs/2006.08832)]
- Human Motion Trajectory Prediction: A Survey, 2019. \[[paper](https://arxiv.org/abs/1905.06113)\]
- A literature review on the prediction of pedestrian behavior in urban scenarios, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569415)\]
- Survey on Vision-Based Path Prediction. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-91131-1_4)\]
- Autonomous vehicles that interact with pedestrians: A survey of theory and practice. \[[paper](https://arxiv.org/abs/1805.11773)\]
- Trajectory data mining: an overview. \[[paper](https://dl.acm.org/citation.cfm?id=2743025)\]
- A survey on motion prediction and risk assessment for intelligent vehicles. \[[paper](https://robomechjournal.springeropen.com/articles/10.1186/s40648-014-0001-z)\]

### Physics Systems with Interaction

- EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2003.13924)\]
- Interaction Templates for Multi-Robot Systems, IROS 2019. \[[paper](https://ieeexplore.ieee.org/abstract/document/8737744/)\]
- Factorised Neural Relational  Inference for Multi-Interaction Systems, ICML workshop 2019. \[[paper](https://arxiv.org/abs/1905.08721v1)\] \[[code](https://github.com/ekwebb/fNRI)\]
- Physics-as-Inverse-Graphics: Joint Unsupervised Learning of Objects and Physics from Video, 2019. \[[paper](https://arxiv.org/pdf/1905.11169v1.pdf)\]
- Neural Relational Inference for Interacting Systems, ICML 2018. \[[paper](https://arxiv.org/abs/1802.04687v2)\] \[[code](https://github.com/ethanfetaya/NRI)\]
- Unsupervised Learning of Latent Physical Properties Using Perception-Prediction Networks, UAI 2018. \[[paper](http://arxiv.org/abs/1807.09244v2)\]
- Relational inductive biases, deep learning, and graph networks, 2018. \[[paper](https://arxiv.org/abs/1806.01261v3)\]
- Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and their Interactions, ICLR 2018. \[[paper](http://arxiv.org/abs/1802.10353v1)\]
- Graph networks as learnable physics engines for inference and control, ICML 2018. \[[paper](http://arxiv.org/abs/1806.01242v1)\]
- Flexible Neural Representation for Physics Prediction, 2018. \[[paper](http://arxiv.org/abs/1806.08047v2)\]
- A simple neural network module for relational reasoning, 2017. \[[paper](http://arxiv.org/abs/1706.01427v1)\]
- VAIN: Attentional Multi-agent Predictive Modeling, NIPS 2017. \[[paper](https://arxiv.org/pdf/1706.06122.pdf)\]
- Visual Interaction Networks, 2017. \[[paper](http://arxiv.org/abs/1706.01433v1)\]
- A Compositional Object-Based Approach to Learning Physical Dynamics, ICLR 2017. \[[paper](http://arxiv.org/abs/1612.00341v2)\]
- Interaction Networks for Learning about Objects, Relations and Physics, 2016. \[[paper](https://arxiv.org/abs/1612.00222)\]\[[code](https://github.com/higgsfield/interaction_network_pytorch)\]

### Intelligent Vehicles & Traffic
- EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2003.13924)\]
- V2VNet- Vehicle-to-Vehicle Communication for Joint Perception and Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2008.07519)]
- SMART- Simultaneous Multi-Agent Recurrent Trajectory Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2007.13078)]
- SimAug- Learning Robust Representations from Simulation for Trajectory Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2004.02022)]
- Learning Lane Graph Representations for Motion Forecasting, ECCV 2020. [[paper](https://arxiv.org/abs/2007.13732)]
- Implicit Latent Variable Model for Scene-Consistent Motion Forecasting, ECCV 2020. [[paper](https://arxiv.org/abs/2007.12036)]
- Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding, ECCV 2020. [[paper](https://arxiv.org/abs/2003.03212)]
- Kernel Trajectory Maps for Multi-Modal Probabilistic Motion Prediction, CoRL 2019. \[[paper](https://arxiv.org/abs/1907.05127)\] \[[code](https://github.com/wzhi/KernelTrajectoryMaps)\]
- Social-WaGDAT: Interaction-aware Trajectory Prediction via Wasserstein Graph Double-Attention Network, 2020. \[[paper](https://arxiv.org/abs/2002.06241)\]
- Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs, 2019. \[[paper](https://arxiv.org/pdf/1912.01118.pdf)\] \[[code](https://gamma.umd.edu/researchdirections/autonomousdriving/spectralcows/)\]
- Joint Prediction for Kinematic Trajectories in Vehicle-Pedestrian-Mixed Scenes, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bi_Joint_Prediction_for_Kinematic_Trajectories_in_Vehicle-Pedestrian-Mixed_Scenes_ICCV_2019_paper.pdf)\]
- Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Thiede_Analyzing_the_Variety_Loss_in_the_Context_of_Probabilistic_Trajectory_ICCV_2019_paper.pdf)\]
- Looking to Relations for Future Trajectory Forecast, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Looking_to_Relations_for_Future_Trajectory_Forecast_ICCV_2019_paper.pdf)\]
- Jointly Learnable Behavior and Trajectory Planning for Self-Driving Vehicles, IROS 2019. \[[paper](https://arxiv.org/abs/1910.04586)\]
- Sharing Is Caring: Socially-Compliant Autonomous Intersection Negotiation, IROS 2019. \[[paper](https://pdfs.semanticscholar.org/f4b2/021353bba52224eb33923b3b98956e2c9821.pdf)\]
- INFER: INtermediate Representations for FuturE PRediction, IROS 2019. \[[paper](https://arxiv.org/abs/1903.10641)\] \[[code](https://github.com/talsperre/INFER)\]
- Deep Predictive Autonomous Driving Using Multi-Agent Joint Trajectory Prediction and Traffic Rules, IROS 2019. \[[paper](http://rllab.snu.ac.kr/publications/papers/2019_iros_predstl.pdf)\]
- NeuroTrajectory: A Neuroevolutionary Approach to Local State Trajectory Learning for Autonomous Vehicles, IROS 2019. \[[paper](https://arxiv.org/abs/1906.10971)\]
- Urban Street Trajectory Prediction with Multi-Class LSTM Networks, IROS 2019. \[N/A\]
- Spatiotemporal Learning of Directional Uncertainty in Urban Environments with Kernel Recurrent Mixture Density Networks, IROS 2019. \[[paper](https://ieeexplore.ieee.org/document/8772158)\]
- Conditional generative neural system for probabilistic trajectory prediction, IROS 2019. \[[paper](https://arxiv.org/abs/1905.01631)\]
- Interaction-aware multi-agent tracking and probabilistic behavior prediction via adversarial learning, ICRA 2019. \[[paper](https://arxiv.org/abs/1904.02390)\]
- Generic Tracking and Probabilistic Prediction Framework and Its Application in Autonomous Driving, IEEE Trans. Intell. Transport. Systems, 2019. \[[paper](https://www.researchgate.net/publication/334560415_Generic_Tracking_and_Probabilistic_Prediction_Framework_and_Its_Application_in_Autonomous_Driving)\]
- Coordination and trajectory prediction for vehicle interactions via bayesian generative modeling, IV 2019. \[[paper](https://arxiv.org/abs/1905.00587)\]
- Wasserstein generative learning with kinematic constraints for probabilistic interactive driving behavior prediction, IV 2019. \[[paper](https://ieeexplore.ieee.org/document/8813783)\]
- GRIP: Graph-based Interaction-aware Trajectory Prediction, ITSC 2019. \[[paper](https://arxiv.org/abs/1907.07792)\]
- AGen: Adaptable Generative Prediction Networks for Autonomous Driving, IV 2019. \[[paper](http://www.cs.cmu.edu/~cliu6/files/iv19-1.pdf)\]
- TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf>)\], \[[code](https://github.com/rohanchandra30/TrackNPred)\]
- Multi-Step Prediction of Occupancy Grid Maps with Recurrent Neural Networks, CVPR 2019. \[[paper](https://arxiv.org/pdf/1812.09395.pdf)\]
- Argoverse: 3D Tracking and Forecasting With Rich Maps, CVPR 2019 \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf)\]
- Robust Aleatoric Modeling for Future Vehicle Localization, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Hudnell_Robust_Aleatoric_Modeling_for_Future_Vehicle_Localization_CVPRW_2019_paper.pdf)\]
- Pedestrian occupancy prediction for autonomous vehicles, IRC 2019. \[paper\]
- Context-based path prediction for targets with switching dynamics, 2019.\[[paper](https://link.springer.com/article/10.1007/s11263-018-1104-4)\]
- Deep Imitative Models for Flexible Inference, Planning, and Control, 2019. \[[paper](https://arxiv.org/abs/1810.06544)\]
- Infer: Intermediate representations for future prediction, 2019. \[[paper](https://arxiv.org/abs/1903.10641)\]\[[code](https://github.com/talsperre/INFER)\]
- Multi-agent tensor fusion for contextual trajectory prediction, 2019. \[[paper](https://arxiv.org/abs/1904.04776)\]
- Context-Aware Pedestrian Motion Prediction In Urban Intersections, 2018. \[[paper](https://arxiv.org/abs/1806.09453)\]
- Generic probabilistic interactive situation recognition and prediction: From virtual to real, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8569780)\]
- Generic vehicle tracking framework capable of handling occlusions based on modified mixture particle filter, IV 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8500626)\]
- Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs, 2018. \[[paper](https://arxiv.org/abs/1805.05499)\]
- Sequence-to-sequence prediction of vehicle trajectory via lstm encoder-decoder architecture, 2018. \[[paper](https://arxiv.org/abs/1802.06338)\]
- R2P2: A ReparameteRized Pushforward Policy for diverse, precise generative path forecasting, ECCV 2018. \[[paper](https://www.cs.cmu.edu/~nrhineha/R2P2.html)\]
- Predicting trajectories of vehicles using large-scale motion priors, IV 2018. \[[paper](https://ieeexplore.ieee.org/document/8500604)\]
- Vehicle trajectory prediction by integrating physics-and maneuver based approaches using interactive multiple models, 2018. \[[paper](https://ieeexplore.ieee.org/document/8186191)\]
- Motion Prediction of Traffic Actors for Autonomous Driving using Deep Convolutional Networks, 2018. \[[paper](https://arxiv.org/abs/1808.05819v1)\]
- Generative multi-agent behavioral cloning, 2018. \[[paper](https://www.semanticscholar.org/paper/Generative-Multi-Agent-Behavioral-Cloning-Zhan-Zheng/ccc196ada6ec9cad1e418d7321b0cd6813d9b261)\]
- Deep Sequence Learning with Auxiliary Information for Traffic Prediction, KDD 2018. \[[paper](https://arxiv.org/pdf/1806.07380.pdf)\], \[[code](https://github.com/JingqingZ/BaiduTraffic)\]
- Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction, 2017. \[[paper](https://link.springer.com/article/10.1007/s10514-017-9619-z)\]
- Probabilistic long-term prediction for autonomous vehicles, IV 2017. \[[paper](https://ieeexplore.ieee.org/abstract/document/7995726)\]
- Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network, ITSC 2017. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
- Desire: Distant future prediction in dynamic scenes with interacting agents, CVPR 2017. \[[paper](https://arxiv.org/abs/1704.04394)\]\[[code](https://github.com/yadrimz/DESIRE)\]
- Imitating driver behavior with generative adversarial networks, 2017. \[[paper](https://arxiv.org/abs/1701.06699)\]\[[code](https://github.com/sisl/gail-driver)\]
- Infogail: Interpretable imitation learning from visual demonstrations, 2017. \[[paper](https://arxiv.org/abs/1703.08840)\]\[[code](https://github.com/YunzhuLi/InfoGAIL)\]
- Long-term planning by short-term prediction, 2017. \[[paper](https://arxiv.org/abs/1602.01580)\]
- Long-term path prediction in urban scenarios using circular distributions, 2017. \[[paper](https://www.sciencedirect.com/science/article/pii/S0262885617301853)\]
- Deep learning driven visual path prediction from a single image, 2016. \[[paper](https://arxiv.org/abs/1601.07265)\]
- Understanding interactions between traffic participants based on learned behaviors, 2016. \[[paper](https://ieeexplore.ieee.org/document/7535554)\]
- Visual path prediction in complex scenes with crowded moving objects, CVPR 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7780661/)\]
- A game-theoretic approach to replanning-aware interactive scene prediction and planning, 2016. \[[paper](https://ieeexplore.ieee.org/document/7353203)\]
- Intention-aware online pomdp planning for autonomous driving in a crowd, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139219)\]
- Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
- Patch to the future: Unsupervised visual prediction, CVPR 2014. \[[paper](http://ieeexplore.ieee.org/abstract/document/6909818/)\]
- Mobile agent trajectory prediction using bayesian nonparametric reachability trees, 2011. \[[paper](https://dspace.mit.edu/handle/1721.1/114899)\]

### Pedestrians

- EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2003.13924)\]
- Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2005.08514)]
- It is not the Journey but the Destination- Endpoint Conditioned Trajectory Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2004.02025)]
- How Can I See My Future? FvTraj: Using First-person View for Pedestrian Trajectory Prediction, ECCV 2020. [[paper](http://graphics.cs.uh.edu/wp-content/papers/2020/2020-ECCV-PedestrianTrajPrediction.pdf)]
- Dynamic and Static Context-aware LSTM for Multi-agent Motion Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2008.00777)]
- Human Trajectory Forecasting in Crowds: A Deep Learning Perspective, 2020. \[[paper](https://arxiv.org/pdf/2007.03639.pdf)\], \[[code](https://github.com/vita-epfl/trajnetplusplusbaselines)\]
- SimAug: Learning Robust Representations from 3D Simulation for Pedestrian Trajectory Prediction in Unseen Cameras, ECCV 2020. \[[paper](https://arxiv.org/pdf/2004.02022)\], \[[code](https://github.com/JunweiLiang/Multiverse)\]
- DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting, ICPR 2020. \[[paper](https://arxiv.org/abs/2005.12661)\] \[[code](https://github.com/alexmonti19/dagnet)\]
- Disentangling Human Dynamics for Pedestrian Locomotion Forecasting with Noisy Supervision, WACV 2020. \[[paper](https://arxiv.org/abs/1911.01138)\]
- Social-WaGDAT: Interaction-aware Trajectory Prediction via Wasserstein Graph Double-Attention Network, 2020. \[[paper](https://arxiv.org/abs/2002.06241)\]
- Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction, CVPR 2020. \[[Paper](<https://arxiv.org/pdf/2002.11927.pdf>)\], \[[Code](<https://github.com/abduallahmohamed/Social-STGCNN/>)\]
- The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction, CVPR 2020. \[[paper](https://arxiv.org/pdf/1912.06445.pdf)\], \[[code/dataset](https://next.cs.cmu.edu/multiverse/index.html)\]
- Disentangling Human Dynamics for Pedestrian Locomotion Forecasting with Noisy Supervision, WACV 2020. \[[paper](https://arxiv.org/abs/1911.01138)\]
- The Trajectron: Probabilistic Multi-Agent Trajectory Modeling With Dynamic Spatiotemporal Graphs, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ivanovic_The_Trajectron_Probabilistic_Multi-Agent_Trajectory_Modeling_With_Dynamic_Spatiotemporal_Graphs_ICCV_2019_paper.pdf)\] \[[code](https://github.com/StanfordASL/Trajectron)\]
- STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf)\] \[[code](https://github.com/huang-xx/STGAT)\]
- Instance-Level Future Motion Estimation in a Single Image Based on Ordinal Regression, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kim_Instance-Level_Future_Motion_Estimation_in_a_Single_Image_Based_on_ICCV_2019_paper.pdf)\]
- Social and Scene-Aware Trajectory Prediction in Crowded Spaces, ICCV workshop 2019. \[[paper](https://arxiv.org/pdf/1909.08840.pdf)\] \[[code](https://github.com/Oghma/sns-lstm/)\]
- Stochastic Sampling Simulation for Pedestrian Trajectory Prediction, IROS 2019. \[[paper](https://arxiv.org/abs/1903.01860)\]
- Long-Term Prediction of Motion Trajectories Using Path Homology Clusters, IROS 2019. \[[paper](http://www.csc.kth.se/~fpokorny/static/publications/carvalho2019a.pdf)\]
- StarNet: Pedestrian Trajectory Prediction Using Deep Neural Network in Star Topology, IROS 2019. \[[paper](https://arxiv.org/pdf/1906.01797.pdf)\]
- Learning Generative Socially-Aware Models of Pedestrian Motion, IROS 2019. \[[paper](https://ieeexplore.ieee.org/abstract/document/8760356/)\]
- Situation-Aware Pedestrian Trajectory Prediction with Spatio-Temporal Attention Model, CVWW 2019. \[[paper](https://arxiv.org/pdf/1902.05437.pdf)\]
- Path predictions using object attributes and semantic environment, VISIGRAPP 2019. \[[paper](http://mprg.jp/data/MPRG/C_group/C20190225_minoura.pdf)\]
- Probabilistic Path Planning using Obstacle Trajectory Prediction, CoDS-COMAD 2019. \[[paper](https://dl.acm.org/citation.cfm?id=3297006)\]
- Human Trajectory Prediction using Adversarial Loss, hEART 2019. \[[paper](http://www.strc.ch/2019/Kothari_Alahi.pdf)\], \[[code](https://github.com/vita-epfl/AdversarialLoss-SGAN)\]
- Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, CVPR 2019. \[[*Precognition Workshop*](https://sites.google.com/view/ieeecvf-cvpr2019-precognition)\], \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Amirian_Social_Ways_Learning_Multi-Modal_Distributions_of_Pedestrian_Trajectories_With_GANs_CVPRW_2019_paper.pdf)\], \[[code](<https://github.com/amiryanj/socialways>)\]
- Peeking into the Future: Predicting Future Person Activities and Locations in Videos, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Peeking_Into_the_Future_Predicting_Future_Person_Activities_and_Locations_CVPR_2019_paper.pdf)\], \[[code](https://github.com/google/next-prediction)\]
- Learning to Infer Relations for Future Trajectory Forecast, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Choi_Learning_to_Infer_Relations_for_Future_Trajectory_Forecast_CVPRW_2019_paper.pdf)\]
- TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf>)\]
- Which Way Are You Going? Imitative Decision Learning for Path Forecasting in Dynamic Scenes, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Which_Way_Are_You_Going_Imitative_Decision_Learning_for_Path_CVPR_2019_paper.pdf>)\]
- Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf>)\]\[[code](https://github.com/lmb-freiburg/Multimodal-Future-Prediction)\]
- Sophie: An attentive gan for predicting paths compliant to social and physical constraints, CVPR 2019. \[[paper](https://arxiv.org/abs/1806.01482)\]\[[code](https://github.com/hindupuravinash/the-gan-zoo/blob/master/README.md)\]
- Pedestrian path, pose, and intention prediction through gaussian process dynamical models and pedestrian activity recognition, 2019. \[[paper](https://ieeexplore.ieee.org/document/8370119/)\]
- Multimodal Interaction-aware Motion Prediction for Autonomous Street Crossing, 2019. \[[paper](https://arxiv.org/abs/1808.06887)\]
- The simpler the better: Constant velocity for pedestrian motion prediction, 2019. \[[paper](https://arxiv.org/abs/1903.07933)\]
- Pedestrian trajectory prediction in extremely crowded scenarios, 2019. \[[paper](https://www.ncbi.nlm.nih.gov/pubmed/30862018)\]
- Srlstm: State refinement for lstm towards pedestrian trajectory prediction, 2019. \[[paper](https://arxiv.org/abs/1903.02793)\]
- Location-velocity attention for pedestrian trajectory prediction, WACV 2019. \[[paper](https://ieeexplore.ieee.org/document/8659060)\]
- Pedestrian Trajectory Prediction in Extremely Crowded Scenarios, Sensors, 2019. \[[paper](https://www.mdpi.com/1424-8220/19/5/1223/pdf)\]
- A data-driven model for interaction-aware pedestrian motion prediction in object cluttered environments, ICRA 2018. \[[paper](https://arxiv.org/abs/1709.08528)\]
- Move, Attend and Predict: An attention-based neural model for people’s movement prediction, Pattern Recognition Letters 2018. \[[paper](https://reader.elsevier.com/reader/sd/pii/S016786551830182X?token=1EF2B664B70D2B0C3ECDD07B6D8B664F5113AEA7533CE5F0B564EF9F4EE90D3CC228CDEB348F79FEB4E8CDCD74D4BA31)\]
- GD-GAN: Generative Adversarial Networks for Trajectory Prediction and Group Detection in Crowds, ACCV 2018, \[[paper](https://arxiv.org/pdf/1812.07667.pdf)\], \[[demo](https://www.youtube.com/watch?v=7cCIC_JIfms)\]
- Ss-lstm: a hierarchical lstm model for pedestrian trajectory prediction, WACV 2018. \[[paper](https://ieeexplore.ieee.org/document/8354239)\]
- Social Attention: Modeling Attention in Human Crowds, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.04689)\]\[[code](https://github.com/TNTant/social_lstm)\]
- Pedestrian prediction by planning using deep neural networks, ICRA 2018. \[[paper](https://arxiv.org/abs/1706.05904)\]
- Joint long-term prediction of human motion using a planning-based social force approach, ICRA 2018. \[[paper](https://iliad-project.eu/publications/2018-2/joint-long-term-prediction-of-human-motion-using-a-planning-based-social-force-approach/)\]
- Human motion prediction under social grouping constraints, IROS 2018. \[[paper](http://iliad-project.eu/publications/2018-2/human-motion-prediction-under-social-grouping-constraints/)\]
- Future Person Localization in First-Person Videos, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yagi_Future_Person_Localization_CVPR_2018_paper.pdf)\]
- Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, CVPR 2018. \[[paper](https://arxiv.org/abs/1803.10892)\]\[[code](https://github.com/agrimgupta92/sgan)\]
- Group LSTM: Group Trajectory Prediction in Crowded Scenarios, ECCV 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-11015-4_18)\]
- Mx-lstm: mixing tracklets and vislets to jointly forecast trajectories and head poses, CVPR 2018. \[[paper](https://arxiv.org/abs/1805.00652)\]
- Intent prediction of pedestrians via motion trajectories using stacked recurrent neural networks, 2018. \[[paper](http://ieeexplore.ieee.org/document/8481390/)\]
- Transferable pedestrian motion prediction models at intersections, 2018. \[[paper](https://arxiv.org/abs/1804.00495)\]
- Probabilistic map-based pedestrian motion prediction taking traffic participants into consideration, 2018. \[[paper](https://ieeexplore.ieee.org/document/8500562)\]
- A Computationally Efficient Model for Pedestrian Motion Prediction, ECC 2018. \[[paper](https://arxiv.org/abs/1803.04702)\]
- Context-aware trajectory prediction, ICPR 2018. \[[paper](https://arxiv.org/abs/1705.02503)\]
- Set-based prediction of pedestrians in urban environments considering formalized traffic rules, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569434)\]
- Building prior knowledge: A markov based pedestrian prediction model using urban environmental data, ICARCV 2018. \[[paper](https://arxiv.org/abs/1809.06045)\]
- Depth Information Guided Crowd Counting for Complex Crowd Scenes, 2018. \[[paper](https://arxiv.org/abs/1803.02256)\]
- Tracking by Prediction: A Deep Generative Model for Mutli-Person Localisation and Tracking, WACV 2018. \[[paper](https://arxiv.org/abs/1803.03347)\]
- “Seeing is Believing”: Pedestrian Trajectory Forecasting Using Visual Frustum of Attention, WACV 2018. \[[paper](https://ieeexplore.ieee.org/document/8354238)\]
- Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Long-Term_On-Board_Prediction_CVPR_2018_paper.pdf)\], \[[code+data](https://github.com/apratimbhattacharyya18/onboard_long_term_prediction)\]
- Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)\], \[[code](https://github.com/ShanghaiTechCVDL/CIDNN)\]
- Walking Ahead: The Headed Social Force Model, 2017. \[[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169734)\]
- Real-time certified probabilistic pedestrian forecasting, 2017. \[[paper](https://ieeexplore.ieee.org/document/7959047)\]
- A multiple-predictor approach to human motion prediction, ICRA 2017. \[[paper](https://ieeexplore.ieee.org/document/7989265)\]
- Forecasting interactive dynamics of pedestrians with fictitious play, CVPR 2017. \[[paper](https://arxiv.org/abs/1604.01431)\]
- Forecast the plausible paths in crowd scenes, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/386)\]
- Bi-prediction: pedestrian trajectory prediction based on bidirectional lstm classification, DICTA 2017. \[[paper](https://ieeexplore.ieee.org/document/8227412/)\]
- Aggressive, Tense or Shy? Identifying Personality Traits from Crowd Videos, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/17)\]
- Natural vision based method for predicting pedestrian behaviour in urban environments, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317848/)\]
- Human Trajectory Prediction using Spatially aware Deep Attention Models, 2017. [[paper](https://arxiv.org/pdf/1705.09436.pdf)\]
- Soft + Hardwired Attention: An LSTM Framework for Human Trajectory Prediction and Abnormal Event Detection, 2017. \[[paper](https://arxiv.org/pdf/1702.05552.pdf)\]
- Forecasting Interactive Dynamics of Pedestrians with Fictitious Play, CVPR 2017. \[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)\]
- Social LSTM: Human trajectory prediction in crowded spaces, CVPR 2016. \[[paper](http://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)\]\[[code](https://github.com/vita-epfl/trajnetplusplusbaselines)\]
- Comparison and evaluation of pedestrian motion models for vehicle safety systems, ITSC 2016. \[[paper](https://ieeexplore.ieee.org/document/7795912)\]
- Age and Group-driven Pedestrian Behaviour: from Observations to Simulations, 2016. \[[paper](https://collective-dynamics.eu/index.php/cod/article/view/A3)\]
- Structural-RNN: Deep learning on spatio-temporal graphs, CVPR 2016. \[[paper](https://arxiv.org/abs/1511.05298)\]\[[code](https://github.com/asheshjain399/RNNexp)\]
- Intent-aware long-term prediction of pedestrian motion, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487409)\]
- Context-based detection of pedestrian crossing intention for autonomous driving in urban environments, IROS 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7759351/)\]
- Novel planning-based algorithms for human motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487505)\]
- Learning social etiquette: Human trajectory understanding in crowded scenes, ECCV 2016. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33)\]\[[code](https://github.com/SajjadMzf/Pedestrian_Datasets_VIS)\]
- GLMP-realtime pedestrian path prediction using global and local movement patterns, ICRA 2016. \[[paper](http://ieeexplore.ieee.org/document/7487768/)\]
- Knowledge transfer for scene-specific motion prediction, ECCV 2016. \[[paper](https://arxiv.org/abs/1603.06987)\]
- STF-RNN: Space Time Features-based Recurrent Neural Network for predicting People Next Location, SSCI 2016. \[[code](https://github.com/mhjabreel/STF-RNN)\]
- Goal-directed pedestrian prediction, ICCV 2015. \[[paper](https://ieeexplore.ieee.org/document/7406377)\]
- Trajectory analysis and prediction for improved pedestrian safety: Integrated framework and evaluations, 2015. \[[paper](https://ieeexplore.ieee.org/document/7225707)\]
- Predicting and recognizing human interactions in public spaces, 2015. \[[paper](https://link.springer.com/article/10.1007/s11554-014-0428-8)\]
- Learning collective crowd behaviors with dynamic pedestrian-agents, 2015. \[[paper](https://link.springer.com/article/10.1007/s11263-014-0735-3)\]
- Modeling spatial-temporal dynamics of human movements for predicting future trajectories, AAAI 2015. \[[paper](https://aaai.org/ocs/index.php/WS/AAAIW15/paper/view/10126)\]
- Unsupervised robot learning to predict person motion, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139254)\]
- A controlled interactive multiple model filter for combined pedestrian intention recognition and path prediction, ITSC 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7313129/)\]
- Real-Time Predictive Modeling and Robust Avoidance of Pedestrians with Uncertain, Changing Intentions, 2014. \[[paper](https://arxiv.org/abs/1405.5581)\]
- Behavior estimation for a complete framework for human motion prediction in crowded environments, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907734)\]
- Pedestrian’s trajectory forecast in public traffic with artificial neural network, ICPR 2014. \[[paper](https://ieeexplore.ieee.org/document/6977417)\]
- Will the pedestrian cross? A study on pedestrian path prediction, 2014. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
- BRVO: Predicting pedestrian trajectories using velocity-space reasoning, 2014. \[[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364914555543)\]
- Context-based pedestrian path prediction, ECCV 2014. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_40)\]
- Pedestrian path prediction using body language traits, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856498/)\]
- Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
- Learning intentions for improved human motion prediction, 2013. \[[paper](https://ieeexplore.ieee.org/document/6766565)\]

### Mobile Robots

- Multimodal probabilistic model-based planning for human-robot interaction, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.09483)\]\[[code](https://github.com/StanfordASL/TrafficWeavingCVAE)\]
- Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning, ICRA 2017. \[[paper](https://arxiv.org/abs/1609.07845)\]
- Augmented dictionary learning for motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487407)\]
- Predicting future agent motions for dynamic environments, ICMLA 2016. \[[paper](https://www.semanticscholar.org/paper/Predicting-Future-Agent-Motions-for-Dynamic-Previtali-Bordallo/2df8179ac7b819bad556b6d185fc2030c40f98fa)\]
- Bayesian intention inference for trajectory prediction with an unknown goal destination, IROS 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7354203/)\]
- Learning to predict trajectories of cooperatively navigating agents, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907442)\]


### Sport Players

- EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2003.13924)\]
- Imitative Non-Autoregressive Modeling for Trajectory Forecasting and Imputation, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_Imitative_Non-Autoregressive_Modeling_for_Trajectory_Forecasting_and_Imputation_CVPR_2020_paper.html)]
- DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting, ICPR 2020. \[[paper](https://arxiv.org/abs/2005.12661)\] \[[code](https://github.com/alexmonti19/dagnet)\]
- Diverse Generation for Multi-Agent Sports Games, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yeh_Diverse_Generation_for_Multi-Agent_Sports_Games_CVPR_2019_paper.html)\]
- Stochastic Prediction of Multi-Agent Interactions from Partial Observations, ICLR 2019. \[[paper](http://arxiv.org/abs/1902.09641v1)\]
- Generating Multi-Agent Trajectories using Programmatic Weak Supervision, ICLR 2019. \[[paper](http://arxiv.org/abs/1803.07612v6)\]
- Generative Multi-Agent Behavioral Cloning, ICML 2018. \[[paper](http://www.stephanzheng.com/pdf/Zhan_Zheng_Lucey_Yue_Generative_Multi_Agent_Behavioral_Cloning.pdf)\]
- Where Will They Go? Predicting Fine-Grained Adversarial Multi-Agent Motion using Conditional Variational Autoencoders, ECCV 2018. \[[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf)\]
- Coordinated Multi-Agent Imitation Learning, ICML 2017. \[[paper](http://arxiv.org/abs/1703.03121v2)\]
- Generating long-term trajectories using deep hierarchical networks, 2017. \[[paper](https://arxiv.org/abs/1706.07138)\]
- Learning Fine-Grained Spatial Models for Dynamic Sports Play Prediction, ICDM 2014. \[[paper](http://www.yisongyue.com/publications/icdm2014_bball_predict.pdf)]
- Generative Modeling of Multimodal Multi-Human Behavior, 2018. \[[paper](https://arxiv.org/pdf/1803.02015.pdf)\]
- What will Happen Next? Forecasting Player Moves in Sports Videos, ICCV 2017, \[[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Felsen_What_Will_Happen_ICCV_2017_paper.pdf)\]

### Benchmark and Evaluation Metrics

- Testing the Safety of Self-driving Vehicles by Simulating Perception and Prediction, ECCV 2020. [[paper](https://arxiv.org/abs/2008.06020)]
- PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction, ICCV 2019. \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)\]
- Towards a fatality-aware benchmark of probabilistic reaction prediction in highly interactive driving scenarios, ITSC 2018. \[[paper](https://arxiv.org/abs/1809.03478)\]
- How good is my prediction? Finding a similarity measure for trajectory prediction evaluation, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317825/)\]
- Trajnet: Towards a benchmark for human trajectory prediction. \[[website](http://trajnet.epfl.ch/)\]

### Others

- Cyclist trajectory prediction using bidirectional recurrent neural networks, AI 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-03991-2_28)\]
- Road infrastructure indicators for trajectory prediction, 2018. \[[paper](https://ieeexplore.ieee.org/document/8500678)\]
- Using road topology to improve cyclist path prediction, 2017. \[[paper](https://ieeexplore.ieee.org/document/7995734/)\]
- Trajectory prediction of cyclists using a physical model and an artificial neural network, 2016. \[[paper](https://ieeexplore.ieee.org/document/7535484/)\]
