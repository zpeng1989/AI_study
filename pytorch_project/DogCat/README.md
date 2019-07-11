训练数据代码：

```
python main.py train
```


训练数据，两种方法，第一种导入参数的方法，第二种写入的方法；

python main.py test --data-root=./data/test  --batch-size=256 --load-path='checkpoints/squeezenet.pth'

其中 

data-root 是训练数据位置
load-path 模型载入路径



