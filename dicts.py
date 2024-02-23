import numpy as np

### option dictionary
l = [ # raw list of options
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
    'God\'s Rebuke',
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
    'Diablic Edict',
    'Firefly',
    'Wings of Icarus',
    'Static Remnant',
    'Crystal Novas',
    'Crown of Ice',
    'Frost Arrows',
    'Powershot', 
    'Wind Blessing',
    'Shadowraze (Near)',
    'Shadowraze (Medium)',
    'Void',
    'Refresh Dice',
    'Super Refresh Dice',
    'Mjollnir',
    'Aegis of the Immortal',
    'Ballista',
    'Eye of Skadi',
    'Gleipnir',
    'Dragon Scale',
    'Arcanist\'s Armor',
    'Overwhelming Blink',
    'Apex',
    'Nullifier',
    'Yasha and Kaya',
    'Brigand\'s Blade',
    'Fusion Rune',
    'Satanic',
    'Bloodthorn',
    'Hand of Midas',
    'Armlet of Mordiggian',
    'Eternal Shroud',
    'Quickening Charm',
    'Enchanted Quiver',
    'Veil of Discord',
    'Dagon',
    'Shiva\'s Guard',
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
    'Fairy\'s Trinket',
    'Spark of Courage',
    'Psychic Headband',
    'Martyr\'s Plate',
    'Mekansm',
    'Helm of the Undying',
    'Magic Lamp',
    'Ehcnated Quiver',
    'Infused Raindrops',
    'Gossamer Cape',
    'Eye of the Vizier',
    'Rising Rocket',
    'Super Rising Rocket'
    ]
option_dict = {}
for i, key in enumerate(l): # populate option dict with options
    option_dict[key] = i+1 # option 0 is reserved for "not available"

# Aliases
option_dict['Shadowraze (Mediur'] = option_dict['Shadowraze (Medium)']
option_dict['Shadowraze (Mediun'] = option_dict['Shadowraze (Medium)']
option_dict['Light Strike'] = option_dict['Light Strike Array']
option_dict['Gossamer'] = option_dict['Gossamer Cape']
option_dict['Shiva\'$ Guard'] = option_dict['Shiva\'s Guard']
option_dict['God\'$ Rebuke'] = option_dict['God\'s Rebuke']
option_dict['Yasha and'] = option_dict['Yasha and Kaya']
option_dict['Arc'] = option_dict['Arc Lightning']
option_dict['Dragon Fame'] = option_dict['Dragon Flame']
option_dict['Arcanist s Armor'] = option_dict['Arcanist\'s Armor']
option_dict['Arcanist $ Armor'] = option_dict['Arcanist\'s Armor']
option_dict['Crown of lce'] = option_dict['Crown of Ice']
option_dict['Divine Rapler'] = option_dict['Divine Rapier']
option_dict['Shiva\'5 Guard'] = option_dict['Shiva\'s Guard']
option_dict['Martyr\' $ Plate'] = option_dict['Shiva\'s Guard']
option_dict['Dark'] = option_dict['Dark Energy']
option_dict['Arcane'] = option_dict['Arcane Ring']
option_dict['Dragon'] = option_dict['Dragon Slave']



### color lookup table
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256): # populate lookUpTable
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0) * 255.0, 0, 255)

### inverse option dictionary
inv_option_dict = {v: k for k, v in option_dict.items()}

hero_dict = {
    0: 'drow'
}



if __name__ == '__main__':
    print(len(l))