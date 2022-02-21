import config

s_saved = 3
i_saved = 3

def rollback():
    return i_saved, s_saved

def update_gec_save(i, s):
    i_saved = i
    s_saved = s
