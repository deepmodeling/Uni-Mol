MODEL_CONFIG = {
    "weight":{
        "protein": "poc_pre_220816.pt",
        "molecule_no_h": "mol_pre_no_h_220816.pt",
        "molecule_all_h": "mol_pre_all_h_220816.pt",
        "crystal": "mp_all_h_230313.pt",
        "oled": "oled_pre_no_h_230101.pt",
    },
    "dict":{
        "protein": "poc.dict.txt",
        "molecule_no_h": "mol.dict.txt",
        "molecule_all_h": "mol.dict.txt",
        "crystal": "mp.dict.txt",
        "oled": "oled.dict.txt",
    },
}

MODEL_CONFIG_V2 = {
    'weight': {
        '84m': 'modelzoo/84M/checkpoint.pt',
        '164m': 'modelzoo/164M/checkpoint.pt',
        '310m': 'modelzoo/310M/checkpoint.pt',
        '570m': 'modelzoo/570M/checkpoint.pt',
        '1.1B': 'modelzoo/1.1B/checkpoint.pt',
    },
}