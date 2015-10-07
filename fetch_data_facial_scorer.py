
# python script to collect images from a preset webpage

import os
try:
    # Python 2 compat
    from urllib2 import Request, build_opener, URLError
except ImportError:
    # Python 3
    from urllib.request import Request, build_opener
    from urllib.error import URLError

import lxml.html
from lxml.etree import ElementTree
import numpy as np

import time
import cPickle as pickle

from contextlib import closing
# you will need to have firefox installed for the script to work, as selenium uses it to access the webpage
from selenium.webdriver import Firefox # pip install selenium
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

url = "http://www.facejoking.com/top/global/0"

html_folder = u'html'
images_folder = u'images'
score_folder = u'scores'

if not os.path.exists(html_folder):
    os.makedirs(html_folder)

if not os.path.exists(images_folder):
    os.makedirs(images_folder)

if not os.path.exists(score_folder):
    os.makedirs(score_folder)

html_filename = os.path.join(html_folder, 'page_source.html')
# check if the html already exists
if not os.path.exists(html_filename):
    # since this webpage is loaded trhough javascrip, we need to command it to load all the elements
    print "Retrieving html..."
    with closing(Firefox()) as browser:
        browser.get(url)
        page_source = ""
        while len(page_source) < len(browser.page_source):
            page_source = browser.page_source
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        page_source = browser.page_source
        # encode the payload explicitly as UTF-8, since it contains chinese characters
        page_source = page_source.encode(encoding='UTF-8',errors='strict')

    html_file = open(html_filename, 'wb')
    html_file.write(page_source)
    html_file.close()
    print "Html retrieved successfully"

# decode the payload explicitly as UTF-8
html_content = open(html_filename).read()
if hasattr(html_content, 'decode'):
    html_content = html_content.decode('utf-8')

# search in the tree to find the elements containing the image url and the score
tree = ElementTree(lxml.html.document_fromstring(html_content))
element = tree.find("//div[@id='ColumnContainer']")
children = list(element)
print "Number of images:", len(children)
image_urls = []
scores = []
for child in children:
    image_element = child.find("./a")
    score_element = child.find(".//div[@class='d-name']")
    image_urls.append(image_element.get('href'))
    text_contents = score_element.text_content().split()
    scores.append(text_contents.pop(-1))

# dump the URLs to a file
images_filename = os.path.join(images_folder, 'all-images-url.pkl')
if not os.path.exists(images_filename):
    images_file = open(images_filename, 'wb')
    pickle.dump(image_urls, images_file)
    images_file.close()
    

# download all the images and create a dictionary with the score attached to each image name
img_scores_dict = {}
score_index = 0
for url in image_urls:
    opener = build_opener()
    image_name = url[url.rfind("/")+1:-4]
    image_filename = os.path.join(images_folder, image_name + '.jpg')
    if not os.path.exists(image_filename):
        print("Downloading %s" % url)
        request = Request(url)
        # request.add_header('User-Agent', 'OpenAnything/1.0')
        error = True
        opened = ""
        max_attempts = 5
        while error and max_attempts>0:
            try:
                opened = opener.open(request)
                img_scores_dict[image_name] = scores[score_index]
                score_index += 1
                error = False
            except URLError as e:
                print(e.reason)
                print "Trying again..."
                max_attempts -= 1

        if max_attempts < 1:
            score_index += 1
            continue
        html_content = opened.read()
        image_file = open(image_filename, 'wb')
        image_file.write(html_content)
        image_file.close()
        print "Download completed!"

data_dict_filename = os.path.join(score_folder, 'data_dict.pkl')
if not os.path.exists(data_dict_filename):
    data_dict_file = open(data_dict_filename, 'wb')
    pickle.dump(img_scores_dict, data_dict_file)
    data_dict_file.close()
