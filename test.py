# coding:UTF-8
# author    :Just_silent
# init time :2021/5/12 15:34
# file      :test.py
# IDE       :PyCharm



def change_dict(d):
    d['a'] = d['a']+1
    if d['a']==2:
        return
    change_dict(d)

if __name__ == '__main__':
    d = {
        'a': 0
    }
    x = change_dict(d)
    a=0