# Interaction-aware Behavior and Trajectory Prediction
![Version](https://img.shields.io/badge/Version-0.1-ff69b4.svg) ![LastUpdated](https://img.shields.io/badge/LastUpdated-2019.07-lightgrey.svg)![Topic](https://img.shields.io/badge/Topic-behavior(trajectory)--prediction-yellow.svg?logo=github) [![HitCount](http://hits.dwyl.io/jiachenli94/Interaction-aware-Trajectory-Prediction.svg)](http://hits.dwyl.io/jiachenli94/Interaction-aware-Trajectory-Prediction)

This is a checklist of state-of-the-art research materials (datasets, blogs, papers and public codes) related to trajectory prediction. Wish it could be helpful for both academia and industry. (Still updating)

Maintainer: [**Jiachen Li**](https://jiachenli94.github.io) (UC Berkeley)

Please feel free to send emails to **jiachen_li@berkeley.edu** for questions, discussion and collaborations.

Also welcome to check the current research in our [**MSC Lab**](https://msc.berkeley.edu/research/autonomous-vehicle.html) at UC Berkeley.

A BAIR blog and survey paper is coming soon!

## Datasets
#### Vehicles and Traffic

- [INTERACTION](http://www.interaction-dataset.com/) (newly released by UC Berkeley MSC Lab)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [HighD](https://www.highd-dataset.com/)
- [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
- [Cyclists](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Tsinghua-Daimler_Cyclist_Detec/tsinghua-daimler_cyclist_detec.html)

#### Pedestrians 

* [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)
* [ETH](http://www.vision.ee.ethz.ch/en/datasets/)
* [VIRAT](http://www.viratdata.org/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
* [ATC](https://irc.atr.jp/crest2010_HRI/ATC_dataset/)
* [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
* [Central Station](http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html)
* [Town Center](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets)
* [Edinburgh](http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/)
* [Cityscapes](https://www.cityscapes-dataset.com/login/)
* [Argoverse](https://www.argoverse.org/)
* [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)
* [TrajNet](http://trajnet.stanford.edu/)

#### Sport Players

- [Football](https://datahub.io/collections/football) 

## Literature and Codes

#### Survey Papers

- Human Motion Trajectory Prediction: A Survey, 2019 \[[paper](https://arxiv.org/abs/1905.06113)\]
- Survey on Vision-Based Path Prediction. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-91131-1_4)\]
- A survey on motion prediction and risk assessment for intelligent vehicles. \[[paper](https://robomechjournal.springeropen.com/articles/10.1186/s40648-014-0001-z)\]
- Autonomous vehicles that interact with pedestrians: A survey of theory and practice. \[[paper](https://arxiv.org/abs/1805.11773)\]
- A literature review on the prediction of pedestrian behavior in urban scenarios, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569415)\]
- Trajectory data mining: an overview. \[[paper](https://dl.acm.org/citation.cfm?id=2743025)\]

#### Intelligent Vehicles & Traffic

- Conditional generative neural system for probabilistic trajectory prediction, IROS 2019. \[[paper](https://arxiv.org/abs/1905.01631)\]
- Interaction-aware multi-agent tracking and probabilistic behavior prediction via adversarial learning, ICRA 2019. \[[paper](https://arxiv.org/abs/1904.02390)\]
- Generic Tracking and Probabilistic Prediction Framework and Its Application in Autonomous Driving, IEEE Trans. Intell. Transport. Systems, 2019. \[[paper](https://www.researchgate.net/publication/334560415_Generic_Tracking_and_Probabilistic_Prediction_Framework_and_Its_Application_in_Autonomous_Driving)\]
- Coordination and trajectory prediction for vehicle interactions via bayesian generative modeling, IV 2019. \[[paper](https://arxiv.org/abs/1905.00587)\]
- Wasserstein generative learning with kinematic constraints for probabilistic interactive driving behavior prediction, IV 2019.
- Generic probabilistic interactive situation recognition and prediction: From virtual to real, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8569780)\]
- Generic vehicle tracking framework capable of handling occlusions based on modified mixture particle filter, IV 2018. \[[paper](https://ieeexplore.ieee.org/abstract/document/8500626)\]
- Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs. \[[paper](https://arxiv.org/abs/1805.05499)\]
- Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction. \[[paper](https://link.springer.com/article/10.1007/s10514-017-9619-z)\]
- Multi-Step Prediction of Occupancy Grid Maps with Recurrent Neural Networks, CVPR 2019. \[[paper](https://arxiv.org/pdf/1812.09395.pdf)\]
- Argoverse: 3D Tracking and Forecasting With Rich Maps, CVPR 2019 \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf)\]
- Robust Aleatoric Modeling for Future Vehicle Localization, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Hudnell_Robust_Aleatoric_Modeling_for_Future_Vehicle_Localization_CVPRW_2019_paper.pdf)\]
- Context-Aware Pedestrian Motion Prediction In Urban Intersections \[[paper](https://arxiv.org/abs/1806.09453)\]
- Probabilistic long-term prediction for autonomous vehicles. \[[paper](https://ieeexplore.ieee.org/abstract/document/7995726)\]
- Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network, ITSC 2017. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
- Desire: Distant future prediction in dynamic scenes with interacting agents, CVPR 2017. \[[paper](https://arxiv.org/abs/1704.04394)\]\[[code](https://github.com/yadrimz/DESIRE)\]
- Sequence-to-sequence prediction of vehicle trajectory via lstm encoder-decoder architecture. \[[paper](https://arxiv.org/abs/1802.06338)\]
- R2P2: A ReparameteRized Pushforward Policy for diverse, precise generative path forecasting, ECCV 2018. \[[paper](https://www.cs.cmu.edu/~nrhineha/R2P2.html)\]
- Long-term planning by short-term prediction. \[[paper](https://arxiv.org/abs/1602.01580)\]
- Predicting trajectories of vehicles using large-scale motion priors. \[[paper](https://ieeexplore.ieee.org/document/8500604)\]
- Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
- Pedestrian occupancy prediction for autonomous vehicles, IRC 2019. \[paper\]
- Vehicle trajectory prediction by integrating physics-and maneuver based approaches using interactive multiple models. \[[paper](https://ieeexplore.ieee.org/document/8186191)\]
- Mobile agent trajectory prediction using bayesian nonparametric reachability trees. \[[paper](https://dspace.mit.edu/handle/1721.1/114899)\]
- A game-theoretic approach to replanning-aware interactive scene prediction and planning. \[[paper](https://ieeexplore.ieee.org/document/7353203)\]
- Intention-aware online pomdp planning for autonomous driving in a crowd, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139219)\]
- Long-term path prediction in urban scenarios using circular distributions. \[[paper](https://www.sciencedirect.com/science/article/pii/S0262885617301853)\]
- Motion Prediction of Traffic Actors for Autonomous Driving using Deep Convolutional Networks. \[[paper](https://arxiv.org/abs/1808.05819v1)\]
- Deep learning driven visual path prediction from a single image. \[[paper](https://arxiv.org/abs/1601.07265)\]
- Context-based path prediction for targets with switching dynamics. \[[paper](https://link.springer.com/article/10.1007/s11263-018-1104-4)\]
- Imitating driver behavior with generative adversarial networks. \[[paper](https://arxiv.org/abs/1701.06699)\]\[[code](https://github.com/sisl/gail-driver)\]
- Understanding interactions between traffic participants based on learned
  behaviors. \[[paper](https://ieeexplore.ieee.org/document/7535554)\]
- Infogail: Interpretable imitation learning from visual demonstrations. \[[paper](https://arxiv.org/abs/1703.08840)\]\[[code](https://github.com/YunzhuLi/InfoGAIL)\]
- Deep Imitative Models for Flexible Inference, Planning, and Control. \[[paper](https://arxiv.org/abs/1810.06544)\]
- Infer: Intermediate representations for future prediction. \[[paper](https://arxiv.org/abs/1903.10641)\]\[[code](https://github.com/talsperre/INFER)\]
- Patch to the future: Unsupervised visual prediction, CVPR 2014. \[[paper](http://ieeexplore.ieee.org/abstract/document/6909818/)\]
- Visual path prediction in complex scenes with crowded moving objects, CVPR 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7780661/)\]
- Generative multi-agent behavioral cloning. \[[paper](https://www.semanticscholar.org/paper/Generative-Multi-Agent-Behavioral-Cloning-Zhan-Zheng/ccc196ada6ec9cad1e418d7321b0cd6813d9b261)\]
- Multi-agent tensor fusion for contextual trajectory prediction. \[[paper](https://arxiv.org/abs/1904.04776)\]
- Deep Sequence Learning with Auxiliary Information for Traffic Prediction, KDD 2018. \[[paper](https://arxiv.org/pdf/1806.07380.pdf)\], \[[code](https://github.com/JingqingZ/BaiduTraffic)\]

#### Mobile Robots

- Bayesian intention inference for trajectory prediction with an unknown goal destination, IROS 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7354203/)\]
- Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning, ICRA 2017. \[[paper](https://arxiv.org/abs/1609.07845)\]
- Augmented dictionary learning for motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487407)\]
- Learning to predict trajectories of cooperatively navigating agents, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907442)\]
- Predicting future agent motions for dynamic environments, ICMLA 2016. \[[paper](https://www.semanticscholar.org/paper/Predicting-Future-Agent-Motions-for-Dynamic-Previtali-Bordallo/2df8179ac7b819bad556b6d185fc2030c40f98fa)\]
- Multimodal probabilistic model-based planning for human-robot interaction, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.09483)\]\[[code](https://github.com/StanfordASL/TrafficWeavingCVAE)\]

#### Pedestrians

- A data-driven model for interaction-aware pedestrian motion prediction in object cluttered environments, ICRA 2018. \[[paper](https://arxiv.org/abs/1709.08528)\]
- Move, Attend and Predict: An attention-based neural model for people’s movement prediction, 2018 Pattern Recognition Letters \[[paper](https://reader.elsevier.com/reader/sd/pii/S016786551830182X?token=1EF2B664B70D2B0C3ECDD07B6D8B664F5113AEA7533CE5F0B564EF9F4EE90D3CC228CDEB348F79FEB4E8CDCD74D4BA31)\]
- Situation-Aware Pedestrian Trajectory Prediction with Spatio-Temporal Attention Model, CVWW 2019. \[[paper](https://arxiv.org/pdf/1902.05437.pdf)\]
- GD-GAN: Generative Adversarial Networks for Trajectory Prediction and Group Detection in Crowds, ACCV 2018, \[[paper](https://arxiv.org/pdf/1812.07667.pdf)\], \[[demo](https://www.youtube.com/watch?v=7cCIC_JIfms)\]
- Path predictions using object attributes and semantic environment, VISIGRAPP 2019. \[[paper](http://mprg.jp/data/MPRG/C_group/C20190225_minoura.pdf)\]
- Probabilistic Path Planning using Obstacle Trajectory Prediction, CoDS-COMAD 2019. \[[paper](https://dl.acm.org/citation.cfm?id=3297006)\]
- Human Trajectory Prediction using Adversarial Loss, 2019. \[[paper](http://www.strc.ch/2019/Kothari_Alahi.pdf)\]
- Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, CVPR 2019. \[[*Precognition Workshop*](https://sites.google.com/view/ieeecvf-cvpr2019-precognition)\], \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Amirian_Social_Ways_Learning_Multi-Modal_Distributions_of_Pedestrian_Trajectories_With_GANs_CVPRW_2019_paper.pdf)\], \[[code](<https://github.com/amiryanj/socialways>)\]
- Peeking into the Future: Predicting Future Person Activities and Locations in Videos, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Peeking_Into_the_Future_Predicting_Future_Person_Activities_and_Locations_CVPR_2019_paper.pdf)\], \[[code](https://github.com/google/next-prediction)\]
- Learning to Infer Relations for Future Trajectory Forecast, CVPR 2019. \[[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Choi_Learning_to_Infer_Relations_for_Future_Trajectory_Forecast_CVPRW_2019_paper.pdf)\]
- TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf>)\]
- Which Way Are You Going? Imitative Decision Learning for Path Forecasting in Dynamic Scenes, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Which_Way_Are_You_Going_Imitative_Decision_Learning_for_Path_CVPR_2019_paper.pdf>)\]
- Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction, CVPR 2019.  \[[paper](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf>)\]
- Ss-lstm: a hierarchical lstm model for pedestrian trajectory prediction, WACV 2018. \[[paper](https://ieeexplore.ieee.org/document/8354239)\]
- Sophie: An attentive gan for predicting paths compliant to social and physical constraints, CVPR 2019. \[[paper](https://arxiv.org/abs/1806.01482)\]\[[code](https://github.com/hindupuravinash/the-gan-zoo/blob/master/README.md)\]
- Social Attention: Modeling Attention in Human Crowds, ICRA 2018. \[[paper](https://arxiv.org/abs/1710.04689)\]\[[code](https://github.com/TNTant/social_lstm)\]
- Goal-directed pedestrian prediction, ICCV 2015. \[[paper](https://ieeexplore.ieee.org/document/7406377)\]
- Pedestrian prediction by planning using deep neural networks, ICRA 2018. \[[paper](https://arxiv.org/abs/1706.05904)\]
- Joint long-term prediction of human motion using a planning-based social force approach, ICRA 2018. \[[paper](https://iliad-project.eu/publications/2018-2/joint-long-term-prediction-of-human-motion-using-a-planning-based-social-force-approach/)\]
- Human motion prediction under social grouping constraints, IROS 2018. \[[paper](http://iliad-project.eu/publications/2018-2/human-motion-prediction-under-social-grouping-constraints/)\]
- Social LSTM: Human trajectory prediction in crowded spaces, CVPR 2016. \[[paper](http://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)\]\[[code](https://github.com/quancore/social-lstm)\]
- Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, CVPR 2018. \[[paper](https://arxiv.org/abs/1803.10892)\]\[[code](https://github.com/agrimgupta92/sgan)\]
- Group LSTM: Group Trajectory Prediction in Crowded Scenarios, ECCV 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-11015-4_18)\]
- Comparison and evaluation of pedestrian motion models for vehicle safety systems, ITSC 2016. \[[paper](https://ieeexplore.ieee.org/document/7795912)\]
- Learning intentions for improved human motion prediction. \[[paper](https://ieeexplore.ieee.org/document/6766565)\]
- Walking Ahead: The Headed Social Force Model. \[[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169734)\]
- Real-Time Predictive Modeling and Robust Avoidance of Pedestrians with Uncertain, Changing Intentions. \[[paper](https://arxiv.org/abs/1405.5581)\]
- Behavior estimation for a complete framework for human motion prediction in crowded environments, ICRA 2014. \[[paper](https://ieeexplore.ieee.org/document/6907734)\]
- Pedestrian’s trajectory forecast in public traffic with artificial neural network, ICPR 2014. \[[paper](https://ieeexplore.ieee.org/document/6977417)\]
- Age and Group-driven Pedestrian Behaviour: from Observations to Simulations. \[[paper](https://collective-dynamics.eu/index.php/cod/article/view/A3)\]
- Mx-lstm: mixing tracklets and vislets to jointly forecast trajectories and head poses, CVPR 2018. \[[paper](https://arxiv.org/abs/1805.00652)\]
- Real-time certified probabilistic pedestrian forecasting. \[[paper](https://ieeexplore.ieee.org/document/7959047)\]
- Structural-RNN: Deep learning on spatio-temporal graphs, CVPR 2016. \[[paper](https://arxiv.org/abs/1511.05298)\]\[[code](https://github.com/asheshjain399/RNNexp)\]
- Intent-aware long-term prediction of pedestrian motion, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487409)\]
- Will the pedestrian cross? A study on pedestrian path prediction. \[[paper](https://ieeexplore.ieee.org/document/6632960)\]
- BRVO: Predicting pedestrian trajectories using velocity-space reasoning. \[[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364914555543)\]
- Context-based pedestrian path prediction, ECCV 2014. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_40)\]
- A multiple-predictor approach to human motion prediction, ICRA 2017. \[[paper](https://ieeexplore.ieee.org/document/7989265)\]
- Forecasting interactive dynamics of pedestrians with fictitious play, CVPR 2017. \[[paper](https://arxiv.org/abs/1604.01431)\]
- Pedestrian path, pose, and intention prediction through gaussian process dynamical models and pedestrian activity recognition. \[[paper](https://ieeexplore.ieee.org/document/8370119/)\]
- Trajectory analysis and prediction for improved pedestrian safety: Integrated framework and evaluations. \[[paper](https://ieeexplore.ieee.org/document/7225707)\]
- Predicting and recognizing human interactions in public spaces. \[[paper](https://link.springer.com/article/10.1007/s11554-014-0428-8)\]
- Multimodal Interaction-aware Motion Prediction for Autonomous Street Crossing. \[[paper](https://arxiv.org/abs/1808.06887)\]
- Pedestrian path prediction using body language traits. \[[paper](https://ieeexplore.ieee.org/document/6856498/)\]
- Intent prediction of pedestrians via motion trajectories using stacked recurrent
  neural networks. \[[paper](http://ieeexplore.ieee.org/document/8481390/)\]
- Context-based detection of pedestrian crossing intention for autonomous driving in urban environments, IROS 2016. \[[paper](https://ieeexplore.ieee.org/abstract/document/7759351/)\]
- The simpler the better: Constant velocity for pedestrian motion prediction. \[[paper](https://arxiv.org/abs/1903.07933)\]
- Transferable pedestrian motion prediction models at intersections. \[[paper](https://arxiv.org/abs/1804.00495)\]
- Pedestrian trajectory prediction in extremely crowded scenarios. \[[paper](https://www.ncbi.nlm.nih.gov/pubmed/30862018)\]
- Forecast the plausible paths in crowd scenes, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/386)\]
- Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression. \[[paper](https://ieeexplore.ieee.org/document/6856480)\]
- Novel planning-based algorithms for human motion prediction, ICRA 2016. \[[paper](https://ieeexplore.ieee.org/document/7487505)\]
- Learning collective crowd behaviors with dynamic pedestrian-agents. \[[paper](https://link.springer.com/article/10.1007/s11263-014-0735-3)\]
- Srlstm: State refinement for lstm towards pedestrian trajectory prediction. \[[paper](https://arxiv.org/abs/1903.02793)\]
- Location-velocity attention for pedestrian trajectory prediction, WACV 2019. \[[paper](https://ieeexplore.ieee.org/document/8659060)\]
- Bi-prediction: pedestrian trajectory prediction based on bidirectional lstm
  classification, DICTA 2017. \[[paper](https://ieeexplore.ieee.org/document/8227412/)\]
- Modeling spatial-temporal dynamics of human movements for predicting future trajectories, AAAI 2015. \[[paper](https://aaai.org/ocs/index.php/WS/AAAIW15/paper/view/10126)\]
- Probabilistic map-based pedestrian motion prediction taking traffic participants into consideration. \[[paper](https://ieeexplore.ieee.org/document/8500562)\]
- Unsupervised robot learning to predict person motion, ICRA 2015. \[[paper](https://ieeexplore.ieee.org/document/7139254)\]
- A controlled interactive multiple model filter for combined pedestrian intention
  recognition and path prediction, ITSC 2015. \[[paper](http://ieeexplore.ieee.org/abstract/document/7313129/)\]
- Learning social etiquette: Human trajectory understanding in crowded
  scenes, ECCV 2016. \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33)\]\[[code](https://github.com/SajjadMzf/Pedestrian_Datasets_VIS)\]
- A Computationally Efficient Model for Pedestrian Motion Prediction, ECC 2018. \[[paper](https://arxiv.org/abs/1803.04702)\]
- GLMP-realtime pedestrian path prediction using global and local movement patterns, ICRA 2016. \[[paper](http://ieeexplore.ieee.org/document/7487768/)\]
- Aggressive, Tense or Shy? Identifying Personality Traits from Crowd Videos, IJCAI 2017. \[[paper](https://www.ijcai.org/proceedings/2017/17)\]
- Knowledge transfer for scene-specific motion prediction, ECCV 2016. \[[paper](https://arxiv.org/abs/1603.06987)\]
- Context-aware trajectory prediction, ICPR 2018. \[[paper](https://arxiv.org/abs/1705.02503)\]
- Set-based prediction of pedestrians in urban environments considering formalized traffic rules, ITSC 2018. \[[paper](https://ieeexplore.ieee.org/document/8569434)\]
- Natural vision based method for predicting pedestrian behaviour in urban environments, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317848/)\]
- Building prior knowledge: A markov based pedestrian prediction model using urban environmental data, ICARCV 2018. \[[paper](https://arxiv.org/abs/1809.06045)\]
- Pedestrian Trajectory Prediction in Extremely Crowded Scenarios, Sensors, 2019. \[[paper](https://www.mdpi.com/1424-8220/19/5/1223/pdf)\]
- Depth Information Guided Crowd Counting for Complex Crowd Scenes, 2018.
- Tracking by Prediction: A Deep Generative Model for Mutli-Person Localisation and Tracking, WACV 2018.
- “Seeing is Believing”: Pedestrian Trajectory Forecasting Using Visual Frustum of Attention, WACV 2018.
- Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Long-Term_On-Board_Prediction_CVPR_2018_paper.pdf)\], \[[code+data](https://github.com/apratimbhattacharyya18/onboard_long_term_prediction)\]
- Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction, CVPR 2018. \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)\], \[[code](https://github.com/ShanghaiTechCVDL/CIDNN)\]
- Human Trajectory Prediction using Spatially aware Deep Attention Models, 2017. [[paper](https://arxiv.org/pdf/1705.09436.pdf)\]
- Soft + Hardwired Attention: An LSTM Framework for Human Trajectory Prediction and Abnormal Event Detection, 2017. \[[paper](https://arxiv.org/pdf/1702.05552.pdf)\]
- Forecasting Interactive Dynamics of Pedestrians with Fictitious Play, CVPR 2017. \[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)\]
- STF-RNN: Space Time Features-based Recurrent Neural Network for predicting People Next Location, SSCI 2016. \[[code](https://github.com/mhjabreel/STF-RNN)\]

#### Sport Players

- Generating long-term trajectories using deep hierarchical networks. \[[paper](https://arxiv.org/abs/1706.07138)\]

#### Benchmark and Evaluation Metrics

- Towards a fatality-aware benchmark of probabilistic reaction prediction in highly interactive driving scenarios, ITSC 2018. \[[paper](https://arxiv.org/abs/1809.03478)\]
- Trajnet: Towards a benchmark for human trajectory prediction. \[[website](http://trajnet.epfl.ch/)\]
- How good is my prediction? Finding a similarity measure for trajectory prediction evaluation, ITSC 2017. \[[paper](http://ieeexplore.ieee.org/document/8317825/)\]

#### Others

- Using road topology to improve cyclist path prediction. \[[paper](https://ieeexplore.ieee.org/document/7995734/)\]
- Cyclist trajectory prediction using bidirectional recurrent neural networks, AI 2018. \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-03991-2_28)\]
- Trajectory prediction of cyclists using a physical model and an artificial neural network. \[[paper](https://ieeexplore.ieee.org/document/7535484/)\]
- Road infrastructure indicators for trajectory prediction. \[[paper](https://ieeexplore.ieee.org/document/8500678)\]