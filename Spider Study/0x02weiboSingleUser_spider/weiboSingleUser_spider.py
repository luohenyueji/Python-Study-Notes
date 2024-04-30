# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:08:58 2020
微博单个用户博文信息爬取
@author: luohenyueji
"""

from urllib.parse import urlencode
import requests
from pyquery import PyQuery as pq
import json

# 定义了base_url来表示请求的URL的前半部分
base_url = 'https://m.weibo.cn/api/container/getIndex?'

headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}


# ----- 获取页面
# 微博链接改版，需要输入since_id
# since_id查找见
# https://github.com/Python3WebSpider/WeiboList/issues/9
def get_page(since_id):
    # type,value,containerid为固定参数
    params = {
        'type': 'uid',
        'value': '2830678474',
        'containerid': '1076032830678474',
        'since_id': since_id
    }
    url = base_url + urlencode(params)
    try:
        # 访问页面
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.ConnectionError as e:
        print('Error', e.args)


# ----- 解析网页
def parse_page(jsondata):
    if jsondata:
        items = jsondata.get('data').get('cards')
        for item in items:
            item = item.get('mblog')
            weibo = {}
            # 微博id
            weibo['id'] = item.get('id')
            # 正文
            weibo['text'] = pq(item.get('text')).text()
            # 获赞数
            weibo['attitudes'] = item.get('attitudes_count')
            # 评论数
            weibo['comments'] = item.get('comments_count')
            # 转发数
            weibo['reposts'] = item.get('reposts_count')
            yield weibo


# ----- 保存数据为json文件
def write_to_json(content):
    with open('result.txt', 'a', encoding='utf-8') as f:
        # ensure_ascii不将中文格式化
        f.write(json.dumps(content, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    since_id = 0
    # 爬取页数
    max_page = 10
    for page in range(1, max_page + 1):
        jsondata = get_page(since_id)
        since_id = jsondata.get('data').get('cardlistInfo').get('since_id')
        for i, weibo in enumerate(parse_page(jsondata)):
            print(weibo)
            write_to_json(weibo)
