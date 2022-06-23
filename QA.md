# Question & Answer

### Comparison with OTOC [1]

**Q:** There is also some exploration about the weakly-supervised 3D semantic segmentation via one click on one thing, for example, OTOC [1].
  
OTOC shares several key ideas with SegGroup: 

1. Seg-Level annotation and supervision;
2. Iterative self-learning with pseudo labels;
3. Graph-based label propagation.

According to the reported numbers of OTOC and SegGroup, I expect that the performance of OTOC surpasses the one of SegGroup in terms of absolute numbers. Taking the shared ideas and the performance gap into consideration, a fair comparison and an honest discussion are required.

**A:** This paper OTOC [1] proposes a weakly supervised point cloud semantic segmentation method by annotating one point per instance in the point cloud scene. The method adopts a self-training strategy to iteratively conduct the semantic segmentation model training and pseudo label generation in a graph structure. A relation network is added to generate the per-category prototype and the similarity among graph nodes to optimize the generated pseudo labels. The graph structure is constructed on the over-segmentation results of the point cloud scene, and each graph node represents a segment.

There are two advantages of our method. The first advantage is that our method can be generally applied to semantic and instance segmentation tasks. The second advantage is that our method uses fewer network parameters. We include the comparison in our manuscript on pages 10-11.

##### 1. OTOC is only designed for semantic segmentation, which cannot be directly employed to the instance segmentation task.

This limitation is determined by its objective function (energy function) for pseudo label generation. OTOC [1] minimizes the below energy function to optimize the pseudo labels:
$$ E(Y|V) = \sum_{j}\varphi_{u}(y_j|V,\Theta) + \sum_{j<j'}\varphi_{p}(y_j,y_{j'}|V, \mathcal{R}, \Theta) $$
where $\Theta$ is the semantic segmentation model, $\mathcal{R}$ is the relation network, and $y_j\in\{1,\dots,K\}$ represents the semantic label for the $j$-th segment. The semantic class number is $K$. The unary term $\varphi_{u}(y_j|V,\Theta)$ represents the semantic prediction on the $j$-th segment $v_j\in V$ pooled from the point-level semantic predictions of the semantic segmentation model. The pairwise term $\varphi_{p}(y_j,y_{j'})$ represents the semantic similarity between the $j$-th and $j'$-th segments. In the energy function, both the unary term and the pairwise term only consider semantic labels. The energy function is built upon the point-level semantic prediction outputs of the semantic segmentation model. The final semantic pseudo labels are obtained by minimizing the energy function.
  
Unlike the semantic labels with a predefined class number and class-specific information, the instance IDs are unordered and the number of the IDs is unknown before label generation. For point cloud instance segmentation, the instance labels are either aggregated from point-level semantic predictions or generated from the 3D bounding box detection results. For each predicted instance, the instance segmentation model cannot output the instance prediction scores for all instance IDs in the scene.
Therefore, the energy function can only be adopted for semantic labels.
  
Instead, our method can be generally applied to semantic and instance segmentation tasks. Although we also use the graph structure which is built on over-segmentation results to propagate labels, we design a clustering algorithm to gradually spread semantic and instance labels around each labeled segment. Because the labeled segments contain instance IDs, instance information can also be propagated through clustering.
  
##### 2. The design of our method has the advantage of fewer computational costs.

The parameter number of OTOC is very large, since it adopts two SparseConvNet [2] networks separately for the semantic segmentation model and the relation network. The total parameter number of OTOC is $30.11{\rm M} + 30.11{\rm M} = 60.21{\rm M}$. In our method, the network for the pseudo label generation is light-weight, and the network parameter number is $0.15{\rm M}$. Combining with the semantic segmentation model KPConv [3] and Minkowski [4], the total network parameter number is $0.15{\rm M} + 14.97{\rm M} = 15.12{\rm M}$ and $0.15{\rm M} + 37.85{\rm M} = 38.00{\rm M}$. When we remove the semantic segmentation model and remain the backbone for pseudo label generation, we find the parameter numbers are $30.11{\rm M}$ v.s. $0.15{\rm M}$ for OTOC and our SegGroup. Therefore, our method uses much fewer network parameters than OTOC.

&nbsp;

### Over-segmentation

**Q:** The whole process relies on the preliminary over-segmentation. The method used and the corresponding parameters have to be detailed. What happens if a segment overlaps two different ground-truth objects? I assume that the majority label is kept, but it is not stated in the paper. Does this event occur? If not, it can be said. If yes, the statistics would be interesting to present.

**A:** In this paper, we conduct experiments on the ScanNet [5] dataset and follow the over-segmentation results provided by the dataset. The over-segmentation results are obtained by a normal-based graph cut method [6,7], with the segmentation cluster threshold $kThresh = 0.01$ and the minimum number of vertices per-segment $segMinVerts = 20$.
As segments are very small in this process, in most cases, each segment only contains one single object. We observe that in very few cases one segment may overlap different objects. However, the ground-truth strong labels of the ScanNet dataset are also annotated based on over-segmentation to accelerate the annotation process. Therefore, it is an intrinsic issue of this dataset that may have a minority of incorrect ground-truth labels, both in training and evaluation. The follow-up works have shown that this issue does not significantly affect the performance, and it is an interesting future work to further clean these incorrect labels.

&nbsp;

### Annotation Rule

**Q:** For Seg-Level annotation, the authors ask the annotators to click on the point on the most representative segment of the instance, such as the largest segment of an instance or the most central segment. How do the authors determine the rule to identify the most representative segment? Is there some literature or statistical result to support this annotation rule? Please clarify it.

**A:** Before manual labeling starts, we first generated various types of weak labels from ground-truth strong labels and conducted experiments to compare their performance in our manuscript in Table VIII. We find the performance mainly relies on the labeled segment sizes, i.e., the larger the better. During the annotation process, we ask the annotator to label the largest segment of each instance to represent it. 
The experimental results of Table VIII in our manuscript show that our manual annotation can label most of the largest segments.
According to our observation, the largest segment of each instance is usually the most central one, so we consider it can represent the location of the instance. 

&nbsp;

### References

[1] Z. Liu, X. Qi, and C.-W. Fu. One thing one click: A self-training approach for weakly supervised 3D semantic segmentation. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), pages 1726–1736, 2021.

[2] B. Graham, M. Engelcke, and L. Van Der Maaten. 3D semantic segmentation with submanifold sparse convolutional networks. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), pages 9224–9232, 2018.

[3] H. Thomas, C. R. Qi, J.-E. Deschaud, B. Marcotegui, F. Goulette, and L. J. Guibas. KPConv: Flexible and deformable convolution for point clouds. In Int. Conf. Comput. Vis. (CVPR), pages 6411–6420, 2019.

[4] C. Choy, J. Gwak, and S. Savarese. 4D spatio-temporal convnets: Minkowski convolutional neural networks. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), pages 3075–3084, 2019.

[5] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), pages 5828–5839, 2017.

[6] P. F. Felzenszwalb and D. P. Huttenlocher. Efficient graph-based image segmentation. Int. J. Comput. Vis. (IJCV), 59(2):167–181, 2004.

[7] A. Karpathy, S. Miller, and L. Fei-Fei. Object discovery in 3D scenes via shape analysis. In IEEE Conf. Robotics
Automation, pages 2088–2095, 2013.