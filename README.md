# 模型结构
model\
├── bert_model.py\
├── data_collator.py\
├── data_processor.py\
├── roberta_model.py\
├── trainer.py\
└── utils.py

# 模块功能描述

`bert_model.py`: BERT编码器与三元组预测

重要函数或类：
- `PoolingTripletDecoder`: 三元组解码器，处理池化后的表示
  - `compute_structure_loss`: 结构感知损失计算模块
- `BertPoolingForTripletPrediction`: 主模型，集成BERT编码、池化和三元组预测，支持训练/预测模式切换
- `GatingFusion`: 门控融合单元
- `StructureReconstructor`: 图重构单元

`data_collator.py`: 批处理数据整理

重要函数或类：
- `PoolingCollator`: 将样本转换为模型输入格式，支持训练/预测模式，处理标签和span掩码

`data_processor.py`: 数据加载与特征转换

重要函数或类：
- `DataProcessor`: 基础数据处理（读TSV、获取样本/标签）
- `KGProcessor`: 核心处理器，管理实体/关系/类型约束，构建知识图谱结构，转换样本为模型特征
  - `build_entity_neighbors` 为每个实体缓存邻接实体
  - `convert_examples_to_features` 返回结构信息
- `InputExample/InputFeatures`: 数据容器
- `AlternateDataset`: 数据集包装器
- `_truncate_seq_triple`: 三元组序列截断


`trainer.py`: 训练与采样策略

重要函数或类：
- `GroupRandomSampler/GroupDistributedSampler`: 分组采样器（用于负采样或分布式训练）
- `KGCTrainer`: 训练器，管理数据加载、模型训练、评估和预测流程
  - `_compute_gated_loss`实现门控融合

`utils.py`: 配置参数

重要函数或类：
- `ModelArguments/DataArguments`: 模型和数据超参数

# 训练环境

## 依赖
python=3.7.8 环境依赖:
```plaintext
transformer=3.0.2
tqdm
numpy
```
## 硬件配置
Bert-Base最少需求12GB显存，Bert_Large最少需求24GB显存，我们的实验使用8x NVIDIA V100 SXM2 32G GPUs, 开启分布式数据并行(DDP)并使用fp16精度加快速度。