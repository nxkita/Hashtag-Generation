#References for this model has been gained from John Naujoks for scraping and Alec Morgan 
#for modeling as this was a fairly unfamiliar project for me to work on. 
#Grateful for the opportunity, I enjoyed making this.

import numpy as np
import time
import re
import os
import json
from random import random
from selenium.webdriver import Chrome, Firefox
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.request import urlretrieve
from uuid import uuid4
import boto3
from io import BytesIO
from PIL import Image
import tensorflow as tf


def get_posts(hashtag, n, browser):
    """With the input of an account page, scrape the n most recent posts urls"""
    url = f"https://www.instagram.com/explore/tags/{hashtag}/"
    browser.get(url)
    post = "https://www.instagram.com/p/"
    post_links = []
    images = []
    while len(post_links) < n or len(images) < n:

        img_src = [
            img.get_attribute("src")
            for img in browser.find_elements(By.CSS_SELECTOR, "article img")
        ]
        links = [
            a.get_attribute("href") for a in browser.find_elements(By.TAG_NAME, "a")
        ]

        for link in links:
            if post in link and link not in post_links and len(post_links) < n:
                post_links.append(link)

        for image in img_src:
            if image not in images and len(images) < n:
                images.append(image)

        scroll_down = "window.scrollTo(0, document.body.scrollHeight);"
        browser.execute_script(scroll_down)
        time.sleep(1 + (random() * 5))

    return [
        {"post_link": post_links[i], "image": images[i], "search_hashtag": hashtag}
        for i in range(len(post_links))
    ]


def get_hashtags(url, browser):
    """Return a list of hashtags found in all post's comments"""
    browser.get(url)
    comments_html = browser.find_elements(By.CSS_SELECTOR, "span")
    all_hashtags = []

    for comment in comments_html:
        hashtags = re.findall("#[A-Za-z]+", comment.text)
        if len(hashtags) > 0:
            all_hashtags.extend(hashtags)
    return list(set(all_hashtags))


def get_image(url, hashtag):
    """Download image from given url and return it's name"""
    uuid = uuid4()
    urlretrieve(url, f"data/{hashtag}/{uuid}.jpg")
    name = f"{uuid}.jpg"
    return name


def scrape_data(hashtags, n, delay=5):
    """
    Download n images and return a dictionary with their metadata.
    """
    browser = Firefox()
    
    for hashtag in hashtags: 
        
        posts = get_posts(hashtag, n, browser)
        
        try:
            os.mkdir(f"data/{hashtag}")
        except OSError:
            pass # We probably tried to make something that already exists

        try:
            for post in posts:
                post["hashtags"] = get_hashtags(post["post_link"], browser)
                time.sleep(random() * delay)
                post["image_local_name"] = get_image(post["image"], hashtag)
                time.sleep(random() * delay)
            new_hashtag_metadata = posts
        except:
            new_hashtag_metadata = posts
        
        
        #NOTE TO SELF: transferred code begins here
        if os.path.exists(f"metadata/{hashtag}.json"):
            # We already have metadata for this hashtag, add to it
            with open(f"metadata/{hashtag}.json", "r") as f:
                hashtag_metadata = json.load(f)
                hashtag_metadata += new_hashtag_metadata
        else:
            # We don't have metadata for this hashtag yet, initialize it
            hashtag_metadata = new_hashtag_metadata

        with open(f"metadata/{hashtag}.json", "w") as f:
            json.dump(hashtag_metadata, f)


def prepare_image(img_path, height=160, width=160):
    """Downsample and scale image to prepare it for neural network"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img)
    img = tf.cast(img, tf.float32)
    img = (img/127.5) - 1
    img = tf.image.resize(img, (height, width))
    # Reshape grayscale images to match dimensions of color images
    if img.shape != (160, 160, 3):
        img = tf.concat([img, img, img], axis=2)
    return img


def extract_features(image, neural_network):
    """Return a vector of 1280 deep features for image."""
    image_np = image.numpy()
    images_np = np.expand_dims(image_np, axis=0)
    deep_features = neural_network.predict(images_np)[0]
    return deep_features