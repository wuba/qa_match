# 运行说明
本示例给出了支持一层和两层结构知识库问答运行demo、评测指标及在测试集的效果。

## 数据介绍
[data_demo](../data_demo)所给出的预训练集（pre_train_data），训练集（train_data），验证集(valid_data)，预测集(test_data) 取自58智能问答生产环境下的真实数据，这里仅为了跑通模型，因此只取了少部分数据，其中预训练集9W+（真实场景下数量较大），训练集9W+，验证集和测试集均3000+，具体数据格式可见[README](../README.md)中的数据介绍部分。

## 基于一层结构知识库的自动问答运行示例

### 基于DSSM模型的自动问答

使用DSSM意图匹配模型时，对于一层结构知识库只需要先训练意图匹配模型，然后用训练好的模型对测试集进行预测，最后对意图匹配的结果按照打分阈值高低给出回答类别，当意图匹配打分高于某个阈值时给出唯一回答，当打分低于某个阈值时给出拒识回答，当打分处于这两个阈值之间时给出列表回答。

#### 1.训练DSSM意图匹配模型

```bash
mkdir model && python run_dssm.py --train_path=./data_demo/train_data --valid_path=./data_demo/valid_data --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --softmax_r=45 --embedding_size=256 --learning_rate=0.001 --keep_prob=0.8 --batch_size=250 --num_epoches=30 --negative_size=200 --eval_every=10 --num_units=256 --use_same_cell=True --label2id_path=./model/model_min/min_label2id --vocab2id_path=./model/model_min/min_vocab2id
```

参数说明：

train_path： 训练集

valid_path： 验证集

map_file_path： 领域意图映射文件

model_path： 模型存储路径

softmax_r： 余弦相似度滑动参数

embedding_size： embedding层向量大小

learning_rate： 学习率

keep_prob： dropout过程中keep神经元的概率 

batch_size： batch 大小

num_epoches： epcho个数

negative_size： 负样本数量

eval_every： 每隔多少steps在验证集上检验训练过程中的模型效果

num_units： lstm cell 的单元个数

use_same_cell： 前向后向lstm是否需要用相同的cell（共享一套参数）

label2id_path： <意图,id>映射文件

vocab2id_path： 根据训练数据生成的字典映射文件

#### 2.用意图匹配模型对测试集进行预测

```bash
python dssm_predict.py --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --export_model_dir=./model/model_min/dssm_tf_serving/ --test_data_path=./data_demo/test_data --test_result_path=./model/model_min/result_min_test --softmax_r=45 --batch_size=250 --label2id_file=./model/model_min/min_label2id --vocab2id_file=./model/model_min/min_vocab2id
```

#### 3.意图匹配的结果按照打分阈值高低给出回答类别

```bash
python merge_classifier_match_label.py none ./model/model_min/result_min_test ./data_demo/merge_result_1_level none
```

### 基于轻量级预训练语言模型（SPTM）的自动问答

使用SPTM进行意图匹配时，对于一层结构知识库需要先预训练语言模型，然后基于预训练语言模型与训练集微调意图匹配模型；最终在用训练好的模型对测试集进行预测后，对意图匹配的结果按照打分阈值高低给出回答类别。其阈值判别的方式与无预训练场景相同。

#### 1.预训练语言模型

```bash
cd sptm && mkdir -p model/pretrain && python run_pretraining.py --train_file="../data_demo/pre_train_data" --vocab_file="../data_demo/vocab" --model_save_dir="./model/pretrain" --batch_size=256 --print_step=100 --weight_decay=0 --embedding_dim=1000 --lstm_dim=500 --layer_num=1 --train_step=100000 --warmup_step=1000 --learning_rate=5e-5 --dropout_rate=0.1 --max_predictions_per_seq=10 --clip_norm=1.0 --max_seq_len=100 --use_queue=0
```
参数说明：

vocab：词典文件（需要包含<PAD><UNK><MASK>)

train_file/valid_data：训练/验证集

lstm_dim：lstm的门控单元数

embedding_dim：词嵌入维度

dropout_rate：节点被dropout的比例

layer_num：LSTM的层数

weight_decay：adam的衰减系数

max_predictions_per_seq：每个句子中，最多会mask的词数

clip_norm：梯度裁剪阈值

use_queue：是否使用队列生成预训练数据

#### 2.微调意图匹配模型

注意此处的```init_checkpoint```需要根据预训练的结果进行选取，如没有预训练模型，也可以不填写：

```bash
cd sptm && python run_classifier.py --output_id2label_file="model/id2label.has_init" --vocab_file="../data_demo/vocab" --train_file="../data_demo/train_data" --dev_file="../data_demo/valid_data" --model_save_dir="model/finetune" --lstm_dim=500 --embedding_dim=1000 --opt_type=adam --batch_size=256 --epoch=20 --learning_rate=1e-4 --seed=1 --max_len=100 --print_step=10 --dropout_rate=0.1 --layer_num=1 --init_checkpoint="model/pretrain/lm_pretrain.ckpt-1400"
```
参数说明：

output_id2label_file：（id，标签）映射文件，最后预测的时侯使用

opt_type：优化器类型，有sgd/adagrad/adam几种可选

seed：随机种子的值，使用相同的随机种子保证微调模型结果一致

init_checkpoint：预训练模型保存的checkpoint

#### 3.用意图匹配模型对测试集进行预测

```bash
cd sptm && python run_prediction.py --input_file="../data_demo/test_data" --vocab_file="../data_demo/vocab" --id2label_file="model/id2label.has_init" --model_dir="model/finetune" > "../data_demo/result_pretrain_raw"
```

#### 4.预测结果格式化，按照打分阈值高低给出回答类别

```bash
python pretrain/format_result.py ./data_demo/test_data ./data_demo/result_pretrain_raw ./data_demo/result_pretrain_test
```
参数说明：

argv[1]: 测试集

argv[2]: SPTM打分文件

argv[3]: 格式化SPTM打分文件

```
python merge_classifier_match_label.py none ./data_demo/result_pretrain_test  ./data_demo/merge_result_pretrain none
```

## 基于两层结构知识库的自动问答运行示例
对于两层结构知识库需要先训练领域分类模型和意图匹配模型，然后用训练好的模型对测试集进行预测，最后对领域分类和意图匹配的结果进行融合，给出回答类别，具体融合策略参考README中两层结构知识库问答融合示意图。

### 1.训练LSTM领域分类模型

```bash
mkdir model && python run_bi_lstm.py  --train_path=./data_demo/train_data --valid_path=./data_demo/valid_data --map_file_path=./data_demo/std_data --model_path=./model/model_max --vocab_file=./model/model_max/vocab_max --label_file=./model/model_max/label_max --embedding_size=256 --num_units=256 --batch_size=200 --seq_length=40 --num_epcho=30 --check_every=20 --lstm_layers=2 --lr=0.01 --dropout_keep_prob=0.8
```
参数说明：

train_path： 训练集

valid_path： 验证集

map_file_path： 领域意图映射文件

model_path： 模型存储路径

vocab_file：根据训练数据生成的字典映射文件

label_file：根据训练过程生成的<领域,id>映射文件

embedding_size： embedding层向量大小

num_units： lstm cell 的单元个数

batch_size： batch 大小

seq_length： 参与训练的最大序列长度

num_epcho： epcho个数

check_every： 每隔多少steps在验证集上检验训练过程中的模型效果

lstm_layers： lstm 层数

lr： 学习率

dropout_keep_prob： dropout过程中keep神经元的概率 

### 2.用领域分类模型对测试集进行预测

```bash
python lstm_predict.py --map_file_path=./data_demo/std_data --model_path=./model/model_max --test_data_path=./data_demo/test_data --test_result_path=./model/model_max/result_max_test --batch_size=250 --seq_length=40 --label2id_file=./model/model_max/label_max --vocab2id_file=./model/model_max/vocab_max
```

参数说明：

map_file_path： 领域意图映射文件

model_path： 模型路径

test_data_path： 测试集 

test_result_path： 测试打分结果文件

batch_size： batch 大小

seq_length： 参与训练的最大序列长度（要和训练过程保持一致）

label2id_file：<领域,id>映射文件

vocab2id_file： 根据训练数据生成的字典映射文件

### 3.训练DSSM意图匹配模型
```bash
python run_dssm.py --train_path=./data_demo/train_data --valid_path=./data_demo/valid_data --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --result_file_path=./data/result_min --softmax_r=45 --embedding_size=256 --learning_rate=0.001 --keep_prob=0.8 --batch_size=250 --num_epoches=30 --negative_size=200 --eval_every=10 --num_units=256 --use_same_cell=False --label2id_path=./model/model_min/min_label2id --vocab2id_path=./model/model_min/min_vocab2id
```

### 4.用意图匹配模型对测试集进行预测

```bash
python dssm_predict.py --map_file_path=./data_demo/std_data --model_path=./model/model_min/ --export_model_dir=./model/model_min/dssm_tf_serving/ --test_data_path=./data_demo/test_data --test_result_path=./model/model_min/result_min_test --softmax_r=45 --batch_size=250 --label2id_file=./model/model_min/min_label2id --vocab2id_file=./model/model_min/min_vocab2id
```

### 5.领域分类结果&意图匹配结果进行融合

```bash
python merge_classifier_match_label.py ./model/model_max/result_max_test ./model/model_min/result_min_test ./data_demo/merge_result_2_level ./data_demo/std_data
```

参数说明：

argv[1]: 领域分类打分文件

argv[2]: 意图识别打分文件

argv[3]: 模型融合文件

argv[4]: 领域意图映射文件 

## 模型评测指标及测试集效果

目前qa_match的问答效果评测是基于分类模型的评测，主要看在模型各种回答类型（唯一回答，列表回答，拒绝回答）占比接近真实应回答类型占比下各种回答的类型的准确率、召回率、F1值，具体定义如下：

![评测指标](measurement.png)

对上述一层结构知识库和二层结构知识库示例（数据集具体见[data_demo](../data_demo)）运行后评测效果如下（使用通用深度学习推理服务[dl_inference](https://github.com/wuba/dl_inference)开源项目部署模型来评测推理耗时）：

| 数据集           | 模型                                                         | **唯一回答准确率** | **唯一回答召回率** | **唯一回答**F1 | **CPU**机器上推理耗时 |
| ---------------- | ------------------------------------------------------------ | ------------------ | ------------------ | -------------- | --------------------- |
| 一级知识库数据集 | DSSM[[下载](http://wos.58cdn.com.cn/nOlKjIhGntU/qamatch/model_min.zip)] | 0.8398             | 0.8326             | 0.8362         | **3ms**               |
| 一级知识库数据集 | SPTM[[下载](http://wos.58cdn.com.cn/nOlKjIhGntU/qamatch/model_pretrain.zip)] | 0.8841             | 0.9002             | 0.8921         | 16ms                  |
| 二级知识库数据集 | LSTM+DSSM融合模型[[下载](http://wos.58cdn.com.cn/nOlKjIhGntU/qamatch/model_merge.zip)] | 0.8957             | 0.9027             | **0.8992**     | 18ms                  |

说明：由于示例数据中列表回答真实占比较小，这里我们主要看唯一回答的准确率、召回率和F1值。对于二级知识库数据集，我们也可以使用预训练模型来完成自动问答，这里不做过多描述。
