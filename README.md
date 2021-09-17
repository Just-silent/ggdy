# 内网邮箱-多轮对话

## 目录介绍
```
gddy_system                     通用问答系统
└── chat_bot                    前后端交互
    ├── static
    ├── templates
    ├── app.py                  flask
    data_process                
    ├── get_train_data.py       获取训练数据   
    ├── import_neo4j.py         输入固定格式直接导入
    QA
    ├── response.py             对话系统-获取回答
    train
    ├── evaluator.py            评价simCSE方法的结果
    ├── path.py                 路径集中设置
    ├── similar.py              训练相似度模型
    └── utils.py                工具
```