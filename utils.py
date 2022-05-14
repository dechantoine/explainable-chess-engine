import numpy as np

dict_pieces = {"white": {"R":"rook",
                         "N":"knight",
                         "B":"bishop",
                         "Q":"queen",
                         "K":"king",
                         "P":"pawn"},
               "black": {"r":"rook",
                         "n":"knight",
                         "b":"bishop",
                         "q":"queen",
                         "k":"king",
                         "p":"pawn"}}

def string_to_array(str_board, is_white=True):
    list_board = list(str_board.replace("\n", "").replace(" ", ""))
    if is_white:
        return np.array([np.reshape([1 if p == piece else 0 for p in list_board],
                                    newshape=(8,8)) for piece in list(dict_pieces["white"])])
    else:
        return np.array([np.reshape([1 if p == piece else 0 for p in list_board],
                                newshape=(8,8)) for piece in list(dict_pieces["black"])])