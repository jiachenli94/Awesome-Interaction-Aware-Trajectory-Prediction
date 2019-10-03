# Awesome Interaction-aware Behavior and Trajectory Prediction
![Version](https://img.shields.io/badge/Version-1.0-ff69b4.svg) ![LastUpdated](https://img.shields.io/badge/LastUpdated-2019.07-lightgrey.svg)![Topic](https://img.shields.io/badge/Topic-behavior(trajectory)--prediction-yellow.svg?logo=github) [![HitCount](http://hits.dwyl.io/jiachenli94/Interaction-aware-Trajectory-Prediction.svg)](http://hits.dwyl.io/jiachenli94/Interaction-aware-Trajectory-Prediction)

This is a checklist of state-of-the-art research materials (datasets, blogs, papers and public codes) related to trajectory prediction. Wish it could be helpful for both academia and industry. (Still updating)

**Maintainers**: [**Jiachen Li**](https://jiachenli94.github.io), Hengbo Ma, Jinning Li (University of California, Berkeley)

**Emails**: {jiachen_li, hengbo_ma, jinning_li}@berkeley.edu

Please feel free to pull request to add new resources or send emails to us for questions, discussion and collaborations.

Also welcome to check the current research in our [**MSC Lab**](https://msc.berkeley.edu/research/autonomous-vehicle.html) at UC Berkeley.

A BAIR blog and a survey paper are coming soon!

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
	- [Mobile Robots](#mobile-robots)
	- [Pedestrians](#pedestrians)
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
|           [TRAF](https://gamma.umd.edu/researchdirections/autonomousdriving/trafdataset)            |      Vehicles/buses/cyclists/bikes / people/animals       |           Urban           |     Camera      |

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

### Sport Players

|                       Dataset                       | Agents |   Scenarios    | Sensors |
| :-------------------------------------------------: | :----: | :------------: | :-----: |
| [Football](https://datahub.io/collections/football) | People | Football field | Camera  |

## **Literature and Codes**

### Survey Papers

1. Human Motion Trajectory Prediction: A Survey, 2019 \[[paper](https://arxiv.org/abs/1905.06113)\]
2. A literature review on the prediction of pedestrian behavior in urban scenarios, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569415)\]
3. Survey on Vision-Based Path Prediction. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-91131-1_4)\]
4. Autonomous vehicles that interact with pedestrians: A survey of theory and practice. \[[paper](https://arxiv.org/abs/1805.11773)\]
5. Trajectory data mining: an overview. \[[paper](https://dl.acm.org/citation.cfm?id=2743025)\]
6. A survey on motion prediction and risk assessment for intelligent vehicles. \[[paper](https://robomechjournal.springeropen.com/articles/10.1186/s40648-014-0001-z)\]

### Physics Systems with Interaction

1. Factorised Neural Relational Inference for Multi-Interaction Systems, ICML workshop 2019. \[[paper](https://arxiv.org/abs/1905.08721v1)\] \[[code](https://github.com/ekwebb/fNRI)\]
2. Physics-as-Inverse-Graphics: Joint Unsupervised Learning of Objects and Physics from Video, 2019. \[[paper](https://arxiv.org/pdf/1905.11169v1.pdf)\]
3. Neural Relational Inference for Interacting Systems, ICML 2018. \[[paper](https://arxiv.org/abs/1802.04687v2)\] \[[code](https://github.com/ethanfetaya/NRI)\]
4. Unsupervised Learning of Latent Physical Properties Using Perception-Prediction Networks, UAI 2018. \[[paper](http://arxiv.org/abs/1807.09244v2)\]
5. Relational inductive biases, deep learning, and graph networks, 2018. \[[paper](https://arxiv.org/abs/1806.01261v3)\]
6. Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and their Interactions, ICLR 2018. \[[paper](http://arxiv.org/abs/1802.10353v1)\]
7. Graph networks as learnable physics engines for inference and control, ICML 2018. \[[paper](http://arxiv.org/abs/1806.01242v1)\]
8. Flexible Neural Representation for Physics Prediction, 2018. \[[paper](http://arxiv.org/abs/1806.08047v2)\]
9. A simple neural network module for relational reasoning, 2017. \[[paper](http://arxiv.org/abs/1706.01427v1)\]
10. VAIN: Attentional Multi-agent Predictive Modeling, NIPS 2017. \[[paper](https://arxiv.org/pdf/1706.06122.pdf)\]
11. Visual Interaction Networks, 2017. \[[paper](http://arxiv.org/abs/1706.01433v1)\]
12. A Compositional Object-Based Approach to Learning Physical Dynamics, ICLR 2017. \[[paper](http://arxiv.org/abs/1612.00341v2)\]
13. Interaction Networks for Learning about Objects, Relations and Physics, 2016. \[[paper](https://arxiv.org/abs/1612.00222)\]\[[code](https://github.com/higgsfield/interaction_network_pytorch)\]

### Intelligent Vehicles & Traffic
0. TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf>)\], \[[code](https://github.com/rohanchandra30/TrackNPred)\]
1. Conditional generative neural system for probabilistic trajectory prediction, IROS 2019. \[[paper](https://arxiv.org/abs/1905.01631)\]
2. Interaction-aware multi-agent tracking and probabilistic behavior prediction via adversarial learning, ICRA 2019. \[[paper](https://arxiv.org/abs/1904.02390)\]
3. Generic Tracking and Probabilistic Prediction Framework and Its Application in Autonomous Driving, IEEE Trans. Intell. Transport. Systems, 2019. \[[paper](https://www.researchgate.net/publication/334560415_Generic_Tracking_and_Probabilistic_Prediction_Framework_and_Its_Application_in_Autonomous_Driving)\]
4. Coordination and trajectory prediction for vehicle interactions via bayesian generative modeling, IV 2019. \[[paper](https://arxiv.org/abs/1905.00587)\]
5. Wasserstein generative learning with kinematic constraints for probabilistic interactive driving behavior prediction, IV 2019.
6. AGen: Adaptable Generative Prediction Networks for Autonomous Driving, IV 2019. \[[paper](http://www.cs.cmu.edu/~cliu6/files/iv19-1.pdf)\]
7. Multi-Step Prediction of Occupancy Grid Maps with Recurrent Neural Networks, CVPR 2019. \[[paper](https://arxiv.org/pdf/1812.09395.pdf)\]
8. Argoverse: 3D Tracking and Forecasting With Rich Maps, CVPR 2019 \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf)\]
9. Robust Aleatoric Modeling for Future Vehicle Localization, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Hudnell_Robust_Aleatoric_Modeling_for_Future_Vehicle_Localization_CVPRW_2019_paper.pdf)\]
10. Pedestrian occupancy prediction for autonomous vehicles, IRC 2019. \[paper\]
11. Context-based path prediction for targets with switching dynamics, 2019.\[[paper](https://link.springer.com/article/10.1007/s11263-018-1104-4)\]
12. Deep Imitative Models for Flexible Inference, Planning, and Control, 2019. \[[paper](https://arxiv.org/abs/1810.06544)\]
13. Infer: Intermediate representations for future prediction, 2019. \[[paper](https://arxiv.org/abs/1903.10641)\]\[[code](https://github.com/talsperre/INFER)\]
14. Multi-agent tensor fusion for contextual trajectory prediction, 2019. \[[paper](https://arxiv.org/abs/1904.04776)\]
15. Context-Aware Pedestrian Motion Prediction In Urban Intersections, 2018. \[[paper](https://arxiv.org/abs/1806.09453)\]
16. Generic probabilistic interactive situation recognition and prediction: From virtual to real, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8569780)\]
17. Generic vehicle tracking framework capable of handling occlusions based on modified mixture particle filter, IV 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8500626)\]
18. Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs, 2018. \[[paper](https://arxiv.org/abs/1805.05499)\]
19. Sequence-to-sequence prediction of vehicle trajectory via lstm encoder-decoder architecture, 2018. \[[paper](https://arxiv.org/abs/1802.06338)\]
20. R2P2: A ReparameteRized Pushforward Policy for diverse, precise generative path forecasting, ECCV 2018. \[[paper](https://www.cs.cmu.edu/~nrhineha/R2P2.html)\]
21. Predicting trajectories of vehicles using large-scale motion priors, IV 2018. \[[paper](https://ieeexplore.ieee.org/document/8500604)\]
22. Vehicle trajectory prediction by integrating physics-and maneuver based approaches using interactive multiple models, 2018. \[[paper](https://ieeexplore.ieee.org/document/8186191)\]
23. Motion Prediction of Traffic Actors for Autonomous Driving using Deep Convolutional Networks, 2018. \[[paper](https://arxiv.org/abs/1808.05819v1)\]
24. Generative multi-agent behavioral cloning, 2018. \[[paper](https://www.semanticscholar.org/paper/Generative-Multi-Agent-Behavioral-Cloning-Zhan-Zheng/ccc196ada6ec9cad1e418d7321b0cd6813d9b261)\]
25. Deep Sequence Learning with Auxiliary Information for Traffic Prediction, KDD 2018. \[[paper](https://arxiv.org/pdf/1806.07380.pdf)\], \[[code](https://github.com/JingqingZ/BaiduTraffic)\]
26. Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction, 2017. \[[paper](https://link.springer.com/article/10.1007/s10514-017-9619-z)\]
27. Probabilistic long-term prediction for autonomous vehicles, IV 2017. \[[paper](https://ieeexplore.ieee.org/abstract/document/7995726)\]
28. Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network, ITSC 2017. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
29. Desire: Distant future prediction in dynamic scenes with interacting agents, CVPR 2017. \[[paper](https://arxiv.org/abs/1704.04394)\]\[[code](https://github.com/yadrimz/DESIRE)\]
30. Imitating driver behavior with generative adversarial networks, 2017. \[[paper](https://arxiv.org/abs/1701.06699)\]\[[code](https://github.com/sisl/gail-driver)\]
31. Infogail: Interpretable imitation learning from visual demonstrations, 2017. \[[paper](https://arxiv.org/abs/1703.08840)\]\[[code](https://github.com/YunzhuLi/InfoGAIL)\]
32. Long-term planning by short-term prediction, 2017. \[[paper](https://arxiv.org/abs/1602.01580)\]
33. Long-term path prediction in urban scenarios using circular distributions, 2017. \[[paper](https://www.sciencedirect.com/science/article/pii/S0262885617301853)\]
34. Deep learning driven visual path prediction from a single image, 2016. \[[paper](https://arxiv.org/abs/1601.07265)\]
35. Understanding interactions between traffic participants based on learned behaviors, 2016. \[[paper](https://ieeexplore.ieee.org/document/7535554)\]
36. Visual path prediction in complex scenes with crowded moving objects, CVPR 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7780661/)\]
37. A game-theoretic approach to replanning-aware interactive scene prediction and planning, 2016. \[[paper](https://ieeexplore.ieee.org/document/7353203)\]
38. Intention-aware online pomdp planning for autonomous driving in a crowd, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139219)\]
39. Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
40. Patch to the future: Unsupervised visual prediction, CVPR 2014. \[[paper](http://ieeexplore.ieee.org/abstract/document/6909818/)\]
41. Mobile agent trajectory prediction using bayesian nonparametric reachability trees, 2011. \[[paper](https://dspace.mit.edu/handle/1721.1/114899)\]

### Mobile Robots

1. Multimodal probabilistic model-based planning for human-robot interaction, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.09483)\]\[[code](https://github.com/StanfordASL/TrafficWeavingCVAE)\]
2. Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning, ICRA 2017. \[[paper](https://arxiv.org/abs/1609.07845)\]
3. Augmented dictionary learning for motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487407)\]
4. Predicting future agent motions for dynamic environments, ICMLA 2016. \[[paper](https://www.semanticscholar.org/paper/Predicting-Future-Agent-Motions-for-Dynamic-Previtali-Bordallo/2df8179ac7b819bad556b6d185fc2030c40f98fa)\]
5. Bayesian intention inference for trajectory prediction with an unknown goal destination, IROS 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7354203/)\]
6. Learning to predict trajectories of cooperatively navigating agents, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907442)\]

### Pedestrians

1. Situation-Aware Pedestrian Trajectory Prediction with Spatio-Temporal Attention Model, CVWW 2019. \[[paper](https://arxiv.org/pdf/1902.05437.pdf)\]
2. Path predictions using object attributes and semantic environment, VISIGRAPP 2019. \[[paper](http://mprg.jp/data/MPRG/C_group/C20190225_minoura.pdf)\]
3. Probabilistic Path Planning using Obstacle Trajectory Prediction, CoDS-COMAD 2019. \[[paper](https://dl.acm.org/citation.cfm?id=3297006)\]
4. Human Trajectory Prediction using Adversarial Loss, 2019. \[[paper](http://www.strc.ch/2019/Kothari_Alahi.pdf)\]
5. Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, CVPR 2019. \[[*Precognition Workshop*](https://sites.google.com/view/ieeecvf-cvpr2019-precognition)\], \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Amirian_Social_Ways_Learning_Multi-Modal_Distributions_of_Pedestrian_Trajectories_With_GANs_CVPRW_2019_paper.pdf)\], \[[code](<https://github.com/amiryanj/socialways>)\]
6. Peeking into the Future: Predicting Future Person Activities and Locations in Videos, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Peeking_Into_the_Future_Predicting_Future_Person_Activities_and_Locations_CVPR_2019_paper.pdf)\], \[[code](https://github.com/google/next-prediction)\]
7. Learning to Infer Relations for Future Trajectory Forecast, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Choi_Learning_to_Infer_Relations_for_Future_Trajectory_Forecast_CVPRW_2019_paper.pdf)\]
8. TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf>)\]
9. Which Way Are You Going? Imitative Decision Learning for Path Forecasting in Dynamic Scenes, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Which_Way_Are_You_Going_Imitative_Decision_Learning_for_Path_CVPR_2019_paper.pdf>)\]
10. Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf>)\]
11. Sophie: An attentive gan for predicting paths compliant to social and physical constraints, CVPR 2019. \[[paper](https://arxiv.org/abs/1806.01482)\]\[[code](https://github.com/hindupuravinash/the-gan-zoo/blob/master/README.md)\]
12. Pedestrian path, pose, and intention prediction through gaussian process dynamical models and pedestrian activity recognition, 2019. \[[paper](https://ieeexplore.ieee.org/document/8370119/)\]
13. Multimodal Interaction-aware Motion Prediction for Autonomous Street Crossing, 2019. \[[paper](https://arxiv.org/abs/1808.06887)\]
14. The simpler the better: Constant velocity for pedestrian motion prediction, 2019. \[[paper](https://arxiv.org/abs/1903.07933)\]
15. Pedestrian trajectory prediction in extremely crowded scenarios, 2019. \[[paper](https://www.ncbi.nlm.nih.gov/pubmed/30862018)\]
16. Srlstm: State refinement for lstm towards pedestrian trajectory prediction, 2019. \[[paper](https://arxiv.org/abs/1903.02793)\]
17. Location-velocity attention for pedestrian trajectory prediction, WACV 2019. \[[paper](https://ieeexplore.ieee.org/document/8659060)\]
18. Pedestrian Trajectory Prediction in Extremely Crowded Scenarios, Sensors, 2019. \[[paper](https://www.mdpi.com/1424-8220/19/5/1223/pdf)\]
19. A data-driven model for interaction-aware pedestrian motion prediction in object cluttered environments, ICRA 2018. \[[paper](https://arxiv.org/abs/1709.08528)\]
20. Move, Attend and Predict: An attention-based neural model for people’s movement prediction, Pattern Recognition Letters 2018. \[[paper](https://reader.elsevier.com/reader/sd/pii/S016786551830182X?token=1EF2B664B70D2B0C3ECDD07B6D8B664F5113AEA7533CE5F0B564EF9F4EE90D3CC228CDEB348F79FEB4E8CDCD74D4BA31)\]
21. GD-GAN: Generative Adversarial Networks for Trajectory Prediction and Group Detection in Crowds, ACCV 2018, \[[paper](https://arxiv.org/pdf/1812.07667.pdf)\], \[[demo](https://www.youtube.com/watch?v=7cCIC_JIfms)\]
22. Ss-lstm: a hierarchical lstm model for pedestrian trajectory prediction, WACV 2018. \[[paper](https://ieeexplore.ieee.org/document/8354239)\]
23. Social Attention: Modeling Attention in Human Crowds, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.04689)\]\[[code](https://github.com/TNTant/social_lstm)\]
24. Pedestrian prediction by planning using deep neural networks, ICRA 2018. \[[paper](https://arxiv.org/abs/1706.05904)\]
25. Joint long-term prediction of human motion using a planning-based social force approach, ICRA 2018. \[[paper](https://iliad-project.eu/publications/2018-2/joint-long-term-prediction-of-human-motion-using-a-planning-based-social-force-approach/)\]
26. Human motion prediction under social grouping constraints, IROS 2018. \[[paper](http://iliad-project.eu/publications/2018-2/human-motion-prediction-under-social-grouping-constraints/)\]
27. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, CVPR 2018. \[[paper](https://arxiv.org/abs/1803.10892)\]\[[code](https://github.com/agrimgupta92/sgan)\]
28. Group LSTM: Group Trajectory Prediction in Crowded Scenarios, ECCV 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-11015-4_18)\]
29. Mx-lstm: mixing tracklets and vislets to jointly forecast trajectories and head poses, CVPR 2018. \[[paper](https://arxiv.org/abs/1805.00652)\]
30. Intent prediction of pedestrians via motion trajectories using stacked recurrent neural networks, 2018. \[[paper](http://ieeexplore.ieee.org/document/8481390/)\]
31. Transferable pedestrian motion prediction models at intersections, 2018. \[[paper](https://arxiv.org/abs/1804.00495)\]
32. Probabilistic map-based pedestrian motion prediction taking traffic participants into consideration, 2018. \[[paper](https://ieeexplore.ieee.org/document/8500562)\]
33. A Computationally Efficient Model for Pedestrian Motion Prediction, ECC 2018. \[[paper](https://arxiv.org/abs/1803.04702)\]
34. Context-aware trajectory prediction, ICPR 2018. \[[paper](https://arxiv.org/abs/1705.02503)\]
35. Set-based prediction of pedestrians in urban environments considering formalized traffic rules, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569434)\]
36. Building prior knowledge: A markov based pedestrian prediction model using urban environmental data, ICARCV 2018. \[[paper](https://arxiv.org/abs/1809.06045)\]
37. Depth Information Guided Crowd Counting for Complex Crowd Scenes, 2018. \[[paper](https://arxiv.org/abs/1803.02256)\]
38. Tracking by Prediction: A Deep Generative Model for Mutli-Person Localisation and Tracking, WACV 2018. \[[paper](https://arxiv.org/abs/1803.03347)\]
39. “Seeing is Believing”: Pedestrian Trajectory Forecasting Using Visual Frustum of Attention, WACV 2018. \[[paper](https://ieeexplore.ieee.org/document/8354238)\]
40. Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Long-Term_On-Board_Prediction_CVPR_2018_paper.pdf)\], \[[code+data](https://github.com/apratimbhattacharyya18/onboard_long_term_prediction)\]
41. Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)\], \[[code](https://github.com/ShanghaiTechCVDL/CIDNN)\]
42. Walking Ahead: The Headed Social Force Model, 2017. \[[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169734)\]
43. Real-time certified probabilistic pedestrian forecasting, 2017. \[[paper](https://ieeexplore.ieee.org/document/7959047)\]
44. A multiple-predictor approach to human motion prediction, ICRA 2017. \[[paper](https://ieeexplore.ieee.org/document/7989265)\]
45. Forecasting interactive dynamics of pedestrians with fictitious play, CVPR 2017. \[[paper](https://arxiv.org/abs/1604.01431)\]
46. Forecast the plausible paths in crowd scenes, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/386)\]
47. Bi-prediction: pedestrian trajectory prediction based on bidirectional lstm classification, DICTA 2017. \[[paper](https://ieeexplore.ieee.org/document/8227412/)\]
48. Aggressive, Tense or Shy? Identifying Personality Traits from Crowd Videos, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/17)\]
49. Natural vision based method for predicting pedestrian behaviour in urban environments, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317848/)\]
50. Human Trajectory Prediction using Spatially aware Deep Attention Models, 2017. [[paper](https://arxiv.org/pdf/1705.09436.pdf)\]
51. Soft + Hardwired Attention: An LSTM Framework for Human Trajectory Prediction and Abnormal Event Detection, 2017. \[[paper](https://arxiv.org/pdf/1702.05552.pdf)\]
52. Forecasting Interactive Dynamics of Pedestrians with Fictitious Play, CVPR 2017. \[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)\]
53. Social LSTM: Human trajectory prediction in crowded spaces, CVPR 2016. \[[paper](http://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)\]\[[code](https://github.com/quancore/social-lstm)\]
54. Comparison and evaluation of pedestrian motion models for vehicle safety systems, ITSC 2016. \[[paper](https://ieeexplore.ieee.org/document/7795912)\]
55. Age and Group-driven Pedestrian Behaviour: from Observations to Simulations, 2016. \[[paper](https://collective-dynamics.eu/index.php/cod/article/view/A3)\]
56. Structural-RNN: Deep learning on spatio-temporal graphs, CVPR 2016. \[[paper](https://arxiv.org/abs/1511.05298)\]\[[code](https://github.com/asheshjain399/RNNexp)\]
57. Intent-aware long-term prediction of pedestrian motion, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487409)\]
58. Context-based detection of pedestrian crossing intention for autonomous driving in urban environments, IROS 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7759351/)\]
59. Novel planning-based algorithms for human motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487505)\]
60. Learning social etiquette: Human trajectory understanding in crowded scenes, ECCV 2016. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33)\]\[[code](https://github.com/SajjadMzf/Pedestrian_Datasets_VIS)\]
61. GLMP-realtime pedestrian path prediction using global and local movement patterns, ICRA 2016. \[[paper](http://ieeexplore.ieee.org/document/7487768/)\]
62. Knowledge transfer for scene-specific motion prediction, ECCV 2016. \[[paper](https://arxiv.org/abs/1603.06987)\]
63. STF-RNN: Space Time Features-based Recurrent Neural Network for predicting People Next Location, SSCI 2016. \[[code](https://github.com/mhjabreel/STF-RNN)\]
64. Goal-directed pedestrian prediction, ICCV 2015. \[[paper](https://ieeexplore.ieee.org/document/7406377)\]
65. Trajectory analysis and prediction for improved pedestrian safety: Integrated framework and evaluations, 2015. \[[paper](https://ieeexplore.ieee.org/document/7225707)\]
66. Predicting and recognizing human interactions in public spaces, 2015. \[[paper](https://link.springer.com/article/10.1007/s11554-014-0428-8)\]
67. Learning collective crowd behaviors with dynamic pedestrian-agents, 2015. \[[paper](https://link.springer.com/article/10.1007/s11263-014-0735-3)\]
68. Modeling spatial-temporal dynamics of human movements for predicting future trajectories, AAAI 2015. \[[paper](https://aaai.org/ocs/index.php/WS/AAAIW15/paper/view/10126)\]
69. Unsupervised robot learning to predict person motion, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139254)\]
70. A controlled interactive multiple model filter for combined pedestrian intention recognition and path prediction, ITSC 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7313129/)\]
71. Real-Time Predictive Modeling and Robust Avoidance of Pedestrians with Uncertain, Changing Intentions, 2014. \[[paper](https://arxiv.org/abs/1405.5581)\]
72. Behavior estimation for a complete framework for human motion prediction in crowded environments, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907734)\]
73. Pedestrian’s trajectory forecast in public traffic with artificial neural network, ICPR 2014. \[[paper](https://ieeexplore.ieee.org/document/6977417)\]
74. Will the pedestrian cross? A study on pedestrian path prediction, 2014. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
75. BRVO: Predicting pedestrian trajectories using velocity-space reasoning, 2014. \[[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364914555543)\]
76. Context-based pedestrian path prediction, ECCV 2014. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_40)\]
77. Pedestrian path prediction using body language traits, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856498/)\]
78. Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression, 2014. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
79. Learning intentions for improved human motion prediction, 2013. \[[paper](https://ieeexplore.ieee.org/document/6766565)\]

### Sport Players

1. Diverse Generation for Multi-Agent Sports Games, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yeh_Diverse_Generation_for_Multi-Agent_Sports_Games_CVPR_2019_paper.html)\]
2. Stochastic Prediction of Multi-Agent Interactions from Partial Observations, ICLR 2019. \[[paper](http://arxiv.org/abs/1902.09641v1)\]
3. Generating Multi-Agent Trajectories using Programmatic Weak Supervision, ICLR 2019. \[[paper](http://arxiv.org/abs/1803.07612v6)\]
4. Generative Multi-Agent Behavioral Cloning, ICML 2018. \[[paper](http://www.stephanzheng.com/pdf/Zhan_Zheng_Lucey_Yue_Generative_Multi_Agent_Behavioral_Cloning.pdf)\]
5. Where Will They Go? Predicting Fine-Grained Adversarial Multi-Agent Motion using Conditional Variational Autoencoders, ECCV 2018. \[[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf)\]
6. Coordinated Multi-Agent Imitation Learning, ICML 2017. \[[paper](http://arxiv.org/abs/1703.03121v2)\]
7. Generating long-term trajectories using deep hierarchical networks, 2017. \[[paper](https://arxiv.org/abs/1706.07138)\]
8. Learning Fine-Grained Spatial Models for Dynamic Sports Play Prediction, ICDM 2014. \[[paper](https://ieeexplore.ieee.org/document/7023384/footnotes#footnotes)\]
9. Generative Modeling of Multimodal Multi-Human Behavior, 2018. \[[paper](https://arxiv.org/pdf/1803.02015.pdf)\]

### Benchmark and Evaluation Metrics

1. Towards a fatality-aware benchmark of probabilistic reaction prediction in highly interactive driving scenarios, ITSC 2018. \[[paper](https://arxiv.org/abs/1809.03478)\]
3. How good is my prediction? Finding a similarity measure for trajectory prediction evaluation, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317825/)\]
2. Trajnet: Towards a benchmark for human trajectory prediction. \[[website](http://trajnet.epfl.ch/)\]

### Others

1. Cyclist trajectory prediction using bidirectional recurrent neural networks, AI 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-03991-2_28)\]
2. Road infrastructure indicators for trajectory prediction, 2018. \[[paper](https://ieeexplore.ieee.org/document/8500678)\]
3. Using road topology to improve cyclist path prediction, 2017. \[[paper](https://ieeexplore.ieee.org/document/7995734/)\]
4. Trajectory prediction of cyclists using a physical model and an artificial neural network, 2016. \[[paper](https://ieeexplore.ieee.org/document/7535484/)\]
