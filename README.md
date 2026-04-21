# 2s-AGCN
Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition in CVPR19

# 现代 Foundry 项目层

本仓库现在在 `src/two_stream_agcn` 下提供现代化的项目侧包。它的边界刻意收窄在
2s-AGCN/AAGCN 自身实现上：

- `two_stream_agcn.models`：现代 PyTorch 版 AGCN/AAGCN 单流模型，以及一个轻量双流包装器。
- `two_stream_agcn.data`：原始 split `.npy + .pkl` 布局适配器，输出 Foundry `classification` 任务约定的 `{"inputs": ..., "target": ...}`。
- `two_stream_agcn.checkpoints`：旧 checkpoint 的 best-effort remap / load 辅助函数。
- `two_stream_agcn.integration.register_two_stream_agcn_project()`：显式注册项目侧 model builder 与旧仓数据适配器。

Foundry 继续负责 skeleton 高层实验语义，包括 dataset/protocol 选择、stream/fusion
规则、graph 布局/策略规则、`SkeletonContext`，以及高层 skeleton 配置到
`RunConfig` 的编译。本仓不维护第二套 skeleton 编译器。

`pyproject.toml` 将 Python 固定为 `>=3.12,<3.13`，因为 Foundry 及其关联私有项目
当前不应由 Python 3.13 解析。

# Note

~~PyTorch version should be 0.3! For PyTorch0.4 or higher, the codes need to be modified.~~ \
Now we have updated the code to >=Pytorch0.4. \
A new model named AAGCN is added, which can achieve better performance. 

# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 
        -data\  
          -kinetics_raw\  
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`
     
# Training & Testing

Change the config file depending on what you want.


    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

    `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/test_bone.yaml`

Then combine the generated scores with: 

    `python ensemble.py` --datasets ntu/xview
     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{2sagcn2019cvpr,  
          title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {CVPR},  
          year      = {2019},  
    }
    
    @article{shi_skeleton-based_2019,
        title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
        journal = {arXiv:1912.06971 [cs]},
        author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
        month = dec,
        year = {2019},
	}
# Contact
For any questions, feel free to contact: `lei.shi@nlpr.ia.ac.cn`
