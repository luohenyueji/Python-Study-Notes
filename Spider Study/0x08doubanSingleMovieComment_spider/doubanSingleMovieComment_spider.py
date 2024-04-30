# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 01:08:07 2020
豆瓣单部电影影评爬取与分析
@author: luohenyueji
"""

import requests
from lxml import etree
import time
import random
import jieba
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
from fake_useragent import UserAgent

# 会话维持
session = requests.Session()
# 代理
proxies = {
    "http": "http://113.121.41.26:9999",
}
ua = UserAgent()
# 随机产生UserAgent，定义请求头，防止反爬
# 详情见https://blog.csdn.net/u013421629/article/details/78168430
headers = {'User-Agent': ua.random}


# ----- 登录账号
def login(tologin=True):
    if not tologin:
        return
    url = "https://accounts.douban.com/j/mobile/login/basic"
    data = {
        'name': '豆瓣账号',
        'password': '密码',
        'remember': 'false'
    }
    # 设置代理，从免费代理网站上找出一个可用的代理IP
    user = session.post(url=url, headers=headers, data=data, proxies=proxies)
    print(user.text)


# ----- 根据电影链接获取评论
def spider_lianjie(lianjie):
    page = 0
    f = open('result.txt', 'a+', encoding="utf-8")
    f.seek(0)
    # 从文件指针的地方开始删除内容
    # 结合上句话也就是说清空所有内容
    f.truncate()
    while True:
        comment_url = lianjie[:42] + 'comments'
        params = {
            'start': page * 20,
            'limit': 20,
            'sort': 'new_score',
            'status': 'P'
        }
        html = session.get(url=comment_url, params=params, proxies=proxies, headers=headers)
        page += 1
        print("开始爬取第{0}页***********************************************************************：".format(page))
        print(html.url)
        xpath_tree = etree.HTML(html.text)
        comment_divs = xpath_tree.xpath('//*[@id="comments"]/div')
        if len(comment_divs) > 2:
            # 获取每一条评论的具体内容
            for comment_div in comment_divs:
                comment = comment_div.xpath('./div[2]/p/span/text()')
                if len(comment) > 0:
                    print(comment[0])
                    f.write(comment[0] + '\n')
            time.sleep(int(random.choice([0.5, 0.2, 0.3])))
        else:
            f.close()
            print("大约共{0}页评论".format(page - 1))
            break


# ----- 根据电影id获取评论
def spider_id(movie_id):
    page = 0
    f = open('result.txt', 'a+', encoding='utf-8')
    f.seek(0)
    f.truncate()
    while True:
        # 链接
        movie_url = 'https://movie.douban.com/subject/' + movie_id + '/comments?'
        # 参数
        params = {
            'start': page * 20,
            'limit': 20,
            'sort': 'new_score',
            'status': 'P'
        }
        # 获取数据
        html = session.get(url=movie_url, params=params, proxies=proxies, headers=headers)
        print(html.status_code)
        page += 1
        print("开始爬取第{0}页".format(page))
        print(html.url)
        xpath_tree = etree.HTML(html.text)
        # 评论根节点
        comment_divs = xpath_tree.xpath('//*[@id="comments"]/div')
        if len(comment_divs) > 2:
            # 获取每一条评论的具体内容
            for comment_div in comment_divs:
                comment = comment_div.xpath('./div[2]/p/span/text()')
                # 保存内容
                if len(comment) > 0:
                    print(comment[0])
                    f.write(comment[0] + '\n')
            # 随机停止时间
            time.sleep(int(random.choice([0.5, 0.2, 0.3])))
        else:
            f.close()
            print("大约共{0}页评论".format(page - 1))
            break


# ----- 定义搜索类型
def spider_kind():
    kind = int(input("请选择搜索类型：1.根据电影链接 2.根据电影id："))
    if kind == 1:
        # example: lianjie = 'https://movie.douban.com/subject/30425219/'
        lianjie = input("请输入电影链接：")
        spider_lianjie(lianjie)
    elif kind == 2:
        # example: movie_id = '30425219'
        movie_id = input("请输入电影id：")
        spider_id(movie_id)
    else:
        print("sorry,输入错误！")


# ----- 分割字符
def cut_word():
    with open('result.txt', 'r', encoding='utf-8') as file:
        # 读取文件里面的全部内容
        comment_txt = file.read()
        # 使用jieba进行分割
        wordlist = jieba.cut(comment_txt)
        wl = "/".join(wordlist)
        # print(wl)
        return wl


def create_word_cloud():
    # 设置词云形状图片,numpy+PIL方式读取图片
    # wc_mask = np.array(Image.open('Emile.jpg'))
    # 数据清洗词列表
    stop_words = ['就是', '不是', '但是', '还是', '只是', '这样', '这个', '一个', '什么', '电影', '没有']

    # 设置词云的一些配置，如：字体，背景色，词云形状，大小,生成词云对象
    # 设置font_path避免中文出错
    wc = WordCloud(background_color="white", stopwords=stop_words, max_words=50, scale=4,
                   max_font_size=50, random_state=42, font_path='C:\\Windows\\Font\\simkai.ttf')

    # 设置mask=wc_mask就可以自定义词云形状
    # wc = WordCloud(mask=wc_mask, background_color="white", stopwords=stop_words, max_words=50, scale=4,
    #                max_font_size=50, random_state=42,font_path='C:\\Windows\\Font\\simkai.ttf')    
    # 生成词云
    wl = cut_word()
    # 根据文本生成词云
    wc.generate(wl)

    # 在只设置mask的情况下,你将会得到一个拥有图片形状的词云
    # 开始画图
    plt.imshow(wc, interpolation="bilinear")
    # 为云图去掉坐标轴
    plt.axis("off")
    plt.figure()
    wc.to_file("WordCloud.jpg")
    plt.show()
    

# ----- 生成情感分析表
def data_show():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    f = open('result.txt', 'r', encoding='UTF-8')
    lists = f.readlines()
    sentimentslist = []
    for i in lists:
        # 情感分析
        s = SnowNLP(i)
        sentimentslist.append(s.sentiments)
    print(sentimentslist)
    print(len(sentimentslist))
    plt.hist(sentimentslist, bins=10, facecolor='g')
    plt.xlabel('情感概率')
    plt.ylabel('数量')
    plt.title('情感分析')
    plt.savefig("sentiment.jpg",dpi=300)
    plt.show()



if __name__ == '__main__':
    # 登录账号
    # False表示不登录，不登录的话只能查看前200条评论，登录之后可以查看500条，但登录多了会需要验证码
    login(False)
    # 爬取网页
    spider_kind()
    # 生成词云
    create_word_cloud()
    # 生成情感分析表
    data_show()
