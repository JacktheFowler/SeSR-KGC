# 模型结构
model\
├── bert_model.py BERT  编码器与三元组预测\
├── data_collator.py  批处理数据\
├── data_processor.py 数据加载与特征转换\
├── trainer.py  训练与采样策略\
└── utils.py  配置参数

# 训练

## 依赖
操作系统采用Ubuntu 20.04，windows下请使用wsl2并下载相应Ubuntu镜像。

环境搭建和依赖下载
```sh
conda create -n ml python=3.7.8
conda activate ml
pip install -r requirement.txt
```
如需要fp16推理加速，需要安装apex
```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
三元组分类，以FB13为例
```sh
chmod u+x FB13.sh
./FB13.sh
```
链接推理,以FB15k-237为例
```sh
chmod u+x FB15k-237.sh
./FB15k-237.sh
```
## 网络问题
对于国内用户，请按照算力平台开启学术加速或使用镜像网站如(https://hf-mirror.com)下载本地资源。
```sh
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download bert-based-cache --local-dir bert-base
```
在`utils.py`中`model_name_or_path`修改为
```sh
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./bert-base",  # 注意这里的 ./ 代表当前运行路径
        metadata={"help": "Path to pretrained model"}
    )
```
## 硬件配置
Bert-Base最少需求12GB显存，Bert_Large最少需求24GB显存，我们的实验使用2x NVIDIA RTX 4090 24G GPUs, 开启分布式数据并行(DDP)并使用fp16精度加快速度。
