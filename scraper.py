from selenium import webdriver
import json
import os
import urllib.request as urllib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def get_element(el, count, save_dir):
    try:
        # path and file extension
        url = json.loads(el.get_attribute('innerHTML'))["ou"]
        ext = json.loads(el.get_attribute('innerHTML'))["ity"]
        if not ext in ['jpg','jpeg','png']:
            return
        # raw bytes
        bytez = urllib.urlopen(url, timeout=20).read()

        # save file
        name = os.path.join(save_dir, save_dir+"_tmp_"+str(count)+"."+ext)
        with open(name, "wb") as fd:
            fd.write(bytez)

    except:
        pass
    
def clean_img_names(path):
    img_files = os.listdir(path)
    counter = 0
    for img in img_files:
        ext = img.split('.')[-1]
        base = img.split('_')[0]
        os.rename(os.path.join(path,img), os.path.join(path,base+'_'+str(counter)+'.'+ext))
        counter += 1

def scrape(chromedriver, search, save_dir=None, verbose=True):

    # set save directory to search terms if not given
    if save_dir is None:
        save_dir = search

    # make directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # open browser
    if verbose:
        print("[*] Opening browser")
    browser = webdriver.Chrome(chromedriver)
    browser.get("https://www.google.com/search?q="+search+"&tbm=isch")

    # scroll to capture more images
    if verbose:
        print("[*] Scrolling to generate images")
    [browser.execute_script("window.scrollBy(0,10000)") for i in range(500)]

    # start scraping
    if verbose:
        print("[*] Scraping")
    with ThreadPoolExecutor() as executor:
        for count, el in enumerate(browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')):
            executor.submit(get_element,el,count,save_dir)
            
    browser.close()
    if verbose:
        print("[*] Renaming files")
    clean_img_names(save_dir)