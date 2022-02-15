import config

s_saved = 3
i_saved = 0

def rollback():
    return i_saved, s_saved

def update_save(i, s):
    i_saved = i
    s_saved = s
