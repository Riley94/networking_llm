import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import HyperParameters as H
import Utils as U
import random
import time
import pokemon
import json
import Defining_States
import VR

banned = {'9': [], '8': [], '7': [], '6': [], '5': ['cloyster', 'dugtrio']}
banned_moves = {'5': {'spore', 'sleep powder', 'hypnosis', 'grass whistle', 'sing', 'lovely kiss', 'yawn', 'sleep talk', 'dynamic punch'}}
banned_abilities = {'5': {'sand rush', 'swift swim'}}

def print_team(team):
    print(f"{team[0]['name']}, {team[1]['name']}, {team[2]['name']}")

def form_name(name):
    match name.lower():
        case 'urshifu':
            return "Urshifu-Single-Strike"
        case 'landorus':
            return "Landorus-Incarnate"
        case 'thundurus':
            return "Thundurus-Incarnate"
        case 'tornadus':
            return "Tornadus-Incarnate"
        case 'toxtricity':
            return "Toxtricity-Low-Key"
        case 'aegislash':
            return "Aegislash-Shield"
        case 'gastrodon-east':
            return "Gastrodon"
        case 'darmanitan':
            return 'Darmanitan-Standard'
        case 'darmanitan-galar':
            return 'Darmanitan-Galar-Standard'
        case 'eiscue':
            return 'Eiscue-Ice'
        case 'keldeo':
            return 'keldeo-resolute'
        case 'shaymin':
            return 'shaymin-land'
        case 'meloetta':
            return 'meloetta-aria'
        case 'meowstic':
            return 'meowstic-male'
        case 'florges-white':
            return 'florges'
        case 'Vivillon-Marine':
            return 'vivillon'
        case 'Wishiwashi':
            return 'wishiwashi-school'
        case 'Sinistcha-Masterpiece':
            return 'sinistcha'
        case _:
            return name

def revert_form_name(name):
    match name.lower():
        case 'darmanitan-standard':
            return 'darmanitan'
        case 'darmanitan-galar-standard':
            return 'darmanitan-galar'
        case _:
            return name
    
def load_data():
    teams_list = []
    with open(U.teams, 'r') as f:
        team = []
        error_pokemon = []
        skipped_pokemon = set()
        banned_pokemon = set()
        banned_moves_set = set()
        banned_abilities_set = set()
        banned_items_set = set()
        move_errors = set()
        making_mon = False
        making_team = False
        pkmn = pokemon.Pokemon('', [])
        for line in f:
            try:
                #Making the name and item
                if line[0:3] == "===":
                    making_team = True
                    try:
                        #print(f"{team[0].name}, {team[1].name}, {team[2].name}, {team[3].name}, {team[4].name}, {team[5].name}")
                        if len(team) == 6:
                            #print(team[0].name, team[1].name, team[2].name, team[3].name, team[4].name, team[5].name)
                            teams_list.append(team)
                            if (len(teams_list))%20 == 0:
                                print(f"{len(teams_list)} teams processed")
                            
                        team = []
                    except:
                        pass
                if not making_team:
                    continue

                if "@" in line:
                    making_mon = True
                    pkmn = pokemon.Pokemon('', [])
                    ### Handle incorrect formatted named
                    while line.count('@') > 1: #Handle nicknames with @ in them
                        line = line.replace('@', '', 1)
                    name = line.split('@')[0].replace('(M)', '').replace('(F)', '')
                    if "(" in name:
                        name = name.split('(')[1].replace(')', '')
                    name = name.strip(' ').replace(' ', '-').replace(':', '').replace('.', '')
                    try:
                        if name in banned[H.gen]:
                            team = []
                            making_mon = False
                            making_team = False
                            banned_pokemon.add(name)
                            continue
                    except:
                        pass
                    name = form_name(name)
                    item = line.split('@')[1].strip()
                    ####################################

                    ###Handle pokeapi exceptions:
                    if name.split('-')[0] == 'Silvally':
                        pkmn = pokemon.Pokemon(name.lower(), [name.split('-')[1]])
                    elif name == 'Zygarde':
                        pkmn = pokemon.Pokemon(name.lower(), ['dragon', 'ground'])
                    elif 'Ogerpon-Wellspring' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['grass', 'water'])
                    elif 'Ogerpon-Hearthflame' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['grass', 'fire'])
                    elif 'Tauros-Paldea-Aqua' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['fighting', 'water'])
                    elif 'Oinkologne' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['normal', 'none'])
                    elif 'Mimikyu' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['ghost', 'fairy'])
                    elif 'Vivillon' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['bug', 'flying'])
                    elif 'Enamorus' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['fairy', 'flying'])
                    elif 'Dudunsparce' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['normal', 'none'])
                    elif 'Indeedee' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['normal', 'psychic'])
                    elif 'Maushold' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['normal', 'none'])
                    elif 'Basculegion' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['water', 'ghost'])
                    elif 'Florges' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['fairy', 'none'])
                    elif 'Tauros-Paldea-Blaze' in name:
                        pkmn = pokemon.Pokemon(name.lower(), ['fire', 'fighting'])
                    else: 
                        try:
                            pkmn = pokemon.get_pokemon(name)
                        except:
                            print(name)
                        if pkmn == None:
                            if name not in error_pokemon:
                                try:
                                    pkmn = pokemon.get_pokemon(line.split('@')[0].replace('(M)', '').replace('(F)', '').split('(')[2].replace(')', '').strip(' ').replace(':', '').lower())
                                except:
                                    error_pokemon.append(name)
                                    print(f"Error: {name}")
                    try:
                        if name.lower() not in VR.get_vr():
                            team = []
                            making_mon = False
                            making_team = False
                            skipped_pokemon.add(name)
                            continue
                            
                    except:
                        pass
                    '''print(name)
                    input()'''
                    pkmn.item = item
                    if 'gem' in item.lower() and H.gen == '5':
                        team = []
                        making_mon = False
                        making_team = False
                        banned_items_set.add(name)
                        continue
                    #Finish Name and Item section

                #Handle other pokemon attributes
                elif 'Ability: ' in line:
                    pkmn.ability = line.split('Ability: ')[1].strip()
                    if pkmn.ability.lower() in banned_abilities[H.gen]:
                        team = []
                        making_mon = False
                        making_team = False
                        banned_abilities_set.add(name)
                        #print(f"Banned move: {move}")
                        continue
                elif 'EVs:' in line:
                    pkmn.evs = line.split('EVs: ')[1].strip()
                elif 'Nature' in line:
                    pkmn.nature = line.split(' ')[0].strip()
                elif '-' in line and '=' not in line and len(pkmn.moves) < 4:
                    line = line.replace('-', '', 1)
                    move = line.strip()
                    if move.lower() in banned_moves[H.gen]:
                        team = []
                        making_mon = False
                        making_team = False
                        banned_moves_set.add(name)
                        #print(f"Banned move: {move}")
                        continue
                    pkmn.moves.append(move)
                #################################
                
                elif line.strip() == '' and making_mon:
                    if len(pkmn.moves) > 4:
                        team = []
                        making_mon = False
                        making_team = False
                        move_errors.add(name)
                        continue
                    if len(team) > 0:
                        '''#Don't add teams with dupe typings
                        cur_typings = []
                        for mon in team:
                            for typing in mon.types:
                                cur_typings.append(typing)
                        dupe_typing = False
                        for typing in pkmn.types:
                            if typing in cur_typings and typing != 'none':
                                dupe_typing = True
                                break
                        if not dupe_typing:
                            team.append(pkmn)
                        else:
                            team.append(pkmn)
                            txt = ""
                            for mon in team:
                                txt += f"{mon.name} "
                            print(txt)
                            team = []
                            continue'''
                        team.append(pkmn)
                    else:
                        team.append(pkmn)
                    making_mon = False
            except Exception as e:
                print(f"Error: {line}\n{e}")

        #Clean up last team
        if len(team) == 6:
            
            teams_list.append(team)

    print(f"Errors: {error_pokemon}")
    print(f"Skipped: {skipped_pokemon}")
    print(f"Banned moves: {banned_moves_set}")
    print(f"Banned items: {banned_items_set}")
    print(f"Banned abilities: {banned_abilities_set}")
    print(f"Banned pokemon: {banned_pokemon}")
    print(f"Move Errors: {move_errors}")
    return teams_list


def preprocess_data(teams_text):
    pokemon_data = []
    for team in teams_text:
        try:
            pokemon_data.append([team[0].to_dict(), team[1].to_dict(), team[2].to_dict(), team[3].to_dict(), team[4].to_dict(), team[5].to_dict()])
        except:
            pass
    return pokemon_data

def clean_teams():
    # Save the list of lists directly to a JSON file
    with open(U.processed_teams, 'r') as json_file:
        pokemon_data = json.load(json_file)

    #print(pokemon_data[0])
    '''pokemon_one = pokemon_data[0][0]

    moves = pokemon_one['moves']
    print(pokemon.get_move(moves[0]))'''

    errors = []
    cleaned_teams = []

    for index, team in enumerate(pokemon_data):
        error = False
        for mon in team:
            moves = []
            for move in mon['moves']:
                try:
                    try:
                        if 'Hidden Power' in move:
                            typing = move.split(' ')[2]
                            move = 'hidden-power'
                            cur_move = pokemon.get_move(move)
                            cur_move.name += f"-{typing.lower()}"
                            cur_move.typing = typing
                            moves.append(cur_move.to_dict())
                        else:
                            cur_move = pokemon.get_move(move)
                            moves.append(cur_move.to_dict())
                    except Exception as e:
                        print(f"Error: {index}; {move}; {e}")
                        errors.append(move)
                        error = True
                except Exception as e:
                    print(f"Error: {index}; {move}; {e}")
                    errors.append(move)
                    error = True
            mon['moves'] = moves
            #print(mon['moves'][0])
        if not error:
            cleaned_teams.append(team)
        
        if index%50 == 0:
            print(f"processed {index} teams")

    print(f"Errors: {errors}")
    print(f"Before: {len(pokemon_data)}, After: {len(cleaned_teams)}")

    print("Saving cleaned teams...")
    # Save the list of lists directly to a JSON file
    with open(U.cleaned_teams_loc, 'w') as json_file:
        json.dump(cleaned_teams, json_file, indent=4)
    print("Data cleanup completed successfully.")

def clean_data():
    print('Loading all the data...')
    teams_text_raw = load_data()

    print('Preprocessing data...')
    teams_list = preprocess_data(teams_text_raw)

    print('Saving Teams')
    # Save the list of lists directly to a JSON file
    with open(U.processed_teams, 'w') as json_file:
        json.dump(teams_list, json_file, indent=4)

    print("Cleaning saved teams...")
    clean_teams()

    print("Formatting teams into pytorch Data")
    Defining_States.format_data()

    print("Data has all been formatted and saved!")

if __name__ == "__main__":
    clean_data()
