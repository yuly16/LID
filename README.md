# 语种识别pytorch实现
## 1.Requirement
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install tqdm
```
## 2.Usage
### 1) 数据预处理
* 将普通话音频文件夹放在`LID_data`目录下，并将文件夹命名为`mandarin`
* 将闽南话音频放在`LID_data`目录下，并将该文件夹命名为`hokkien`
* 在`LID`目录下运行
```
python utils/data_preparation.py --base_path DATA_DIR --kaldi_path KALDI_PATH
```       
* 参数说明
    * `DATA_DIR`为`LID_data`所在目录，用绝对路径表示，如`/home/yuly/LID/LID_data`
    * 我们的程序需要用到kaldi语音工具箱。如果没有安装，请先安装kaldi。上面命令的`KALDI_PATH`为`kaldi/egs/sre16/v2`所在的绝对目录，例如`/mnt/workspace2/yuly/kaldi/egs/sre16/v2`
    * 示例：
    ```
    python utils/data_preparation.py --base_path '/home/yuly/LID/LID_data' --kaldi_path '/mnt/workspace2/yuly/kaldi/egs/sre16/v2
    ```
    * 可以看到训练数据文件夹为`dataset_train`，测试数据文件夹为`dataset_test`。
### 2) 训练网络
```
        python train.py --batch_size 512 --epoches 40 --train_dir TRAIN_DIR --test_dir TEST_DIR --is_gpu True
```
* 参数说明
    * `TRAIN_DIR`为`LID_data/dataset_train`文件夹所在的绝对目录，`TEST_DIR`为`LID_data/dataset_test`文件夹所在的绝对目录.
    * 如果运行过程中提示CUDA MEMORY ERROR则可以尝试减小`batch_size`。另外如果想使用cpu进行训练，则将`is_gpu`参数设置为False
* 示例
```
python train.py --batch_size 512 --epoches 40 --train_dir '/home/yuly/LID/LID_data/dataset_train' --test_dir '/home/yuly/LID/LID_data/dataset_test' --is_gpu True
```
* 20个epoch之后趋向于收敛。
* `log/log.txt`中保存每一个epoch的准确率和loss，可以用来绘制loss和acc曲线。

### 3) 测试网络
* 采用自己训练好的模型
```
python test.py --test_dir TEST_DIR --is_gpu True
```
* 采用作者训练好的模型
```
python test.py --test_dir TEST_DIR --model_dir 'log/best_model_0958.pth' --is_gpu True
```