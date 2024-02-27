from multiprocessing import Process,Queue,Pipe
from PIL import Image
from importlib import reload

import dicts
reload(dicts)
from dicts import option_dict, lookUpTable, inv_option_dict, hero_dict

import pytesseract
import cv2
import pyautogui as pag
import numpy as np
import time
import easyocr


global tolerance
tolerance = 1

# global option_dict 

# global reader
reader = easyocr.Reader(['en'])

def quickshow(image):
    # pag.alert('Image Ready!', button = 'OK')
    cv2.namedWindow('pic')
    cv2.moveWindow('pic', 40,30)
    cv2.imshow('pic', image)
    cv2.waitKey(4_000)
    cv2.destroyAllWindows()

def longshow(image):
    # pag.alert('Image Ready!', button = 'OK')
    cv2.namedWindow('pic')
    cv2.moveWindow('pic', 40,30)
    cv2.imshow('pic', image)
    cv2.waitKey(50_000)
    cv2.destroyAllWindows()

def fullscreen():
    pag.click(0,500) # click into application
    return

def recenter():
    fullscreen()
    time.sleep(tolerance)
    pag.typewrite('-', interval=1) # pag.typewrite(['s', '-'], interval=1)
    pag.scroll(-2000)
    return

def select_stage(stage=1, difficulty=1, level=1, speed=1, hero=0):
    recenter()
    time.sleep(tolerance)
    pag.rightClick(1146,381,duration=tolerance) # open stage selection menu
    # pag.click()
    # pag.moveTo(*speed_cords[2])

    stage_coords = {
        1: (600,300),
        2: (900,300),
        3: (1200,300)
    }
    difficulty_coords = {
        1: (625, 450),
        2: (700,450) ,
        3: (750,450),
        4: (800,450),
        5: (900,450)
    }
    level_coords = {
        1: (625, 480),
        2: (700,480) ,
        3: (750,480),
        4: (800,480),
        5: (900,480)
    }
    speed_coords = {
        1: (750,580),
        2: (800,580)
    }
    
    pag.click(*stage_coords[stage], duration = tolerance) # select stage
    pag.click(*difficulty_coords[difficulty], duration = tolerance) # select difficulty
    pag.click(*level_coords[level], duration = tolerance) # select level
    pag.click(*speed_coords[speed], duration = tolerance) # select speed
    pag.click(830, 865, duration = tolerance+0.5) # select confirm
    recenter()
    pag.rightClick(1260, 489, duration = tolerance) # queue for next map
    time.sleep(5)
    pick_hero(hero)

    return stage, difficulty, level, speed, hero

def check_if_choices(template):
    screenshot = pag.screenshot(region=(1600,200, 320, 600)) 
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    region = image[400:]
    pag.alert('region:')
    quickshow(region)
    try:
        pag.locate(template, region)
        pag.alert('Choices found')
        return True
    except:
        pag.alert('Choices not fouind')
        return False

def screenshot():
    screenshot = pag.screenshot(region=(1500,250, 700, 650)) 
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gamma_image = cv2.LUT(image, lookUpTable)
    return gamma_image

def template_match(template, region):
    # pag.alert(cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF))[1] / 1_000_000)
    res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    
    if maxVal > 3_000_000:
        return True
    else:
        False
    
def clean_text(result):
    return ''.join([i for i in result if i not in list('123')]).lstrip().rstrip()

def get_choices(sc=None):
    # fullscreen()
    # time.sleep(1)
    if not sc:
        sc = screenshot()
    image = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY)
    select_region = image[30:-100, 50:-50] 
    equalized_image = cv2.equalizeHist(select_region) # equalize
    # blur = cv2.GaussianBlur(equ, (3,3), 1)
    binary_image = cv2.threshold(select_region,45,255,cv2.THRESH_BINARY)[1] # threshold
    inverted_image = 255 - binary_image # invert
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    processed_image = inverted_image
    # processed_image = 255 - binary_image
    # open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, open_kernel, iterations=1)
    # processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, np.ones((2,2)))

    # pag.alert('processed image')
    # longshow(processed_image)
    
    results = reader.readtext(processed_image)
    
    cleaned_texts = [clean_text(results[i][1]) for i in range(len(results))]
    texts = [text for text in cleaned_texts if text in option_dict]
    print(texts)
    # choices aren't present
    # if len(texts)==0:
    #     print('No Choices Present!')
    #     return False
    # else:
    #     print('Choices detected!')
    
    # pag.alert(texts)

    ret = np.zeros(4)
    for i, text in enumerate(texts):
        ret[i] = option_dict[text]
        
    if (ret[1] == 0 or ret[2] == 0) and ret[0] != 0:
        pag.alert('Ability Missing!')
        quickshow(processed_image)
        pag.alert(cleaned_texts)
        
        pass
    if not ret.any():
        print('No Options Found')
        # pag.alert('Select REgion')
        # longshow(processed_image)
    return ret

def await_choices():
    print('Awaiting Choices')
    choices = get_choices()
    while choices[2] == 0:
        print('Choices not Found')
        choices = get_choices()
    
    print('Returning choices ', choices)
    return choices

def get_reward():
    # Check if Loss
    print(pag.pixel(955, 325))
    if pag.pixelMatchesColor(955, 325, (93, 21, 30), tolerance=4):
        print('Lose!')
        res = -1
    # Check if win
    elif pag.pixelMatchesColor(955, 325, (40, 102, 44), tolerance=4):
        print('Win!')
        res = 1
    else:
        return 0

    return res

def pick_hero(hero):
    recenter()
    hero_template = cv2.imread(f'./heroes/{hero}.png')
    # pag.alert('image ready')
    region = pag.screenshot(region=(80, 230, 400, 600))
    sc = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
    coords = pag.locateOnScreen(hero_template)
    # print(coords)
    pag.click(coords, duration=tolerance) # Select Hero
    pag.click(1700, 1000, duration=tolerance+0.5) # Ready Upw
    return

def reset_game():
    pag.click(950, 800, duration = tolerance)
    time.sleep(tolerance)
    pag.press('=')
    time.sleep(1)
    pag.rightClick(985, 615, duration=tolerance)

if __name__ == '__main__': 
    # print(clean_text('2 Shiva'))
    fullscreen()
    get_choices()
    # select_stage(2, 4, 1, 2, 'drow')
    # reset_game()

    # pag.press(str(1))

    if False: # Pick Stage and Hero
        select_stage(1, 1, 1, 2, 0)
        time.sleep(5)
    
    if False: # Get Choices Once
        print(get_choices())

    if False: # Get Choices Loopas
        while True:
            print(get_choices())
            time.sleep(1)
        
    if False: # get reward
        get_reward()

    if False: # reset game
        reset_game()
    
    