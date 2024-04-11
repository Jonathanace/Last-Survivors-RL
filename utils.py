"""
Stores Ability Names and Populates Name Encoder Dictionary. NEVER Modify the list of names. Except to append
a new choice to the end.

Remember to restart the kernel if you modify this.
"""

from pathlib import Path
import os
import numpy as np
import pyautogui as pag
import cv2

ability_names = [
    'Arc Lightning',
    'Thunder Power',
    'Lightning Bolt',
    'Dragon Slave',
    'Dragon Flame',
    'Light Strike Array',
    'Spirits', 
    'Primal Arcana',
    'Splinter Blast',
    'Poison Attack', 
    'Poison Tooth',
    'Nethertoxin',
    'Proximity Mines',
    'Dark Energy',
    'Aphotic Shield',
    'Shrapnel',
    'Quick Reload',
    'Headshot',
    'Firestorm',
    'Abyssal Stone',
    'Pit of Malice',
    'Spear of Mars',
    'War Shield',
    'Gods Rebuke',
    'Hoof Stomp',
    'Strong Heart',
    'Double Edge',
    'Meat Hook',
    'Fresh Flesh',
    'Rot',
    'Refraction',
    'Secret Psionics',
    'Psi Blades',
    'Mana Shield',
    'Magic Scales',
    'Mystic Snake',
    'Pulse Nova',
    'Time Crystal',
    'Diabolic Edict',
    'Firefly',
    'Wings of Icarus',
    'Static Remnant',
    'Crystal Novas',
    'Crown of Ice',
    'Frost Arrows',
    'Moon Glaives',
    'Darkmoon Totem',
    'Powershot',
    'Wind Blessing',
    'Shadowraze Near',
    'Shadowraze Medium',
    'Void',
    'Unstable Concoction'
]
item_names = [
    'Mjollnir',
    'Aegis of the Immortal',
    'Ballista',
    'Eye of Skadi',
    'Gleipnir',
    'Dragon Scale',
    'Arcanists Armor',
    'Overwhelming Blink',
    'Apex',
    'Nullifier',
    'Yasha and Kaya',
    'Brigands Blade',
    'Fusion Rune',
    'Satanic',
    'Bloodthorn',
    'Hand of Midas',
    'Armlet of Mordiggian',
    'Eternal Shroud',
    'Quickenign Charm',
    'Veil of Discord',
    'Dagon',
    'Shivas Guard',
    'Fluffy Hat',
    'Divine Rapier',
    'Desolator',
    'Ninja Gear',
    'Phase Boots',
    'Paladin Sword',
    'Blade Mail',
    'Vanguard',
    'Extra Gifts',
    'Holy Locket',
    'Tranquil Boots',
    'Arcane Ring',
    'Fairys Trinket',
    'Spark of Courage',
    'Psychic Headband',
    'Martyrs Plate',
    'Mekansm',
    'Helm of the Undying',
    'Magic Lamp',
    'Enchanted Quiver',
    'Infused Raindrops',
    'Gossamer Cape',
    'Eye of the Vizier'
]
other_choice_names = [
    'MAX',
    'Rising Rocket',
    'Super Rising Rocket'
]
namespace = ability_names + item_names + other_choice_names 

icons_dir="./icons"
pathlist = Path(icons_dir).rglob('*')
file_names = {os.path.basename(path)[:-4] for path in pathlist}

num_files = len(namespace)
missing_files = 0
encoder_dict = {}

for i, name in enumerate(namespace):
    encoder_dict[name] = i+1
    if name not in file_names:
        missing_files += 1
        print(f'Warning: {name} has no corresponding image!')
print(f'{num_files-missing_files}/{num_files} images loaded')

def validate_icons():
    missing_files = 0
    for name in namespace:
        if name not in file_names:
            missing_files += 1
            print(f'Warning: {name} has no corresponding image!')
    print(f'{num_files-missing_files}/{num_files} images loaded')

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256): # populate lookUpTable
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0) * 255.0, 0, 255)


def screenshot(save=True, region=(1550,290, 370, 550), filename="screenshot.png"):
    """
    region: a four-integer tuple of the left, top, width, and height of the region to capture
    """
    screenshot = pag.screenshot(region=region) 
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gamma_image = cv2.LUT(image, lookUpTable)
    if save:
        cv2.imwrite(filename, gamma_image) 
    return gamma_image

def quickshow(image):
    # pag.alert('Image Ready!', button = 'OK')
    cv2.namedWindow('pic')
    cv2.moveWindow('pic', 40,30)
    cv2.imshow('pic', image)
    cv2.waitKey(4_000)
    cv2.destroyAllWindows()


def get_choices(image_or_image_path=None, icons_dir="./icons", confidence_threshold=0.70):
    """
    Returns: A list of the names of current choices
    """
    ### Initialize
    if image_or_image_path is None:
        image = cv2.imread("screenshot.png")
    items = {} # rounded y: (name, confidence)
    
    ### Get Choices
    pathlist = Path(icons_dir).rglob('*')
    for path in pathlist:
        if path.is_file():
            file_path = str(path)
            icon = cv2.imread(str(file_path))
            if icon is not None:
                name = os.path.basename(file_path)[:-4]

                # Match Template
                res = cv2.matchTemplate(image, icon, cv2.TM_CCOEFF_NORMED)
                _, confidence, _, loc = cv2.minMaxLoc(res) # min_val, max_val, min_loc, max_loc

                # Resolve Collisions
                if confidence > confidence_threshold:
                    rounded_y = round(loc[1]/100, 1)
                    if rounded_y in items: # collision detected
                        old_conf = items[rounded_y][1]
                        if confidence > old_conf:
                            items[rounded_y] = (name, confidence)
                    else: # no collision
                        items[rounded_y] = (name, confidence)

    ### Convert dictionary to list
    positions = sorted(items.keys())
    sorted_items = [items[position] for position in sorted(positions)]
    
    ### Output Results
    print(f'--- Confidence Threshold: {confidence_threshold} ---')
    print(f'{len(sorted_items )} choices found')
    for item in sorted_items:
        print(item)
    
    return [item[0] for item in sorted_items]


get_choices()