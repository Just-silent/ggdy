import tensorflow.compat.v1 as tf
from flask import Flask, render_template, request
from tensorflow.python.keras.backend import set_session
from QA.response import Response

tf.disable_v2_behavior()

# 程序开始时声明
# sess = tf.Session()
# graph = tf.get_default_graph()

# 在model加载前添加set_session
# set_session(sess)

app = Flask(__name__)
app.static_folder = 'static'

response = Response()

history = {
    'is_right':None,            # None可以判断回答是否正确
    'num_keys':None,            # None可以判断是否完成关键词抽取
    'question_operate':None,    # None可以预测与所推测关键词最相关的问题和答案
    'keys':None,                # None可以推荐关键词
    'response':None,
    'is_asking':0,
    'question':None,
    'last_step':False
}

@app.route("/")
def home():
    return render_template("index.html")

# 该方法只有在输入的时候才能触发
@app.route("/get", methods=['GET'])
def get_bot_response():
    # 无法使用循环，因为return会导致方法的退出
    # 只能使用递归，直到对话结束才退出
    global history
    userText = request.args.get('msg')
    result= response.get_response(userText, history)
    print('问题', userText)
    print('回答', result)
    return result

if __name__ == "__main__":
    app.run(port=7777, debug=False, host='0.0.0.0', threaded=False)

