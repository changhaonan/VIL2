from yacs.config import CfgNode as CN

_C = CN()
_C.DIFFICULTY = 'easy'

_C.EASY = CN()
_C.EASY.BASE_RADIUS_LOW_HIGH = [0.0125, 0.025]
_C.EASY.BASE_LENGTH_LOW_HIGH = [0.2, 0.3]
_C.EASY.PEG_RADIUS_LOW_HIGH = [0.00575, 0.0125]
_C.EASY.PEG_LENGTH_LOW_HIGH = [0.1, 0.15]
_C.EASY.PEG_ANGLE_LOW_HIGH = [40, 50]
_C.EASY.PEG_YAW_LOW_HIGH = [0, 0]
_C.EASY.PEG_BASE_HEIGHT_FRAC_LOW_HIGH = [0.5, 0.6]
_C.EASY.WITH_BOTTOM = False
_C.EASY.BOTTOM_CYLINDER_RADIUS_LOW_HIGH = None
_C.EASY.BOTTOM_CYLINDER_LENGTH_LOW_HIGH = None
_C.EASY.BOTTOM_BOX_SIDE_LOW_HIGH = None
_C.EASY.BOTTOM_BOX_HEIGHT_LOW_HIGH = None
_C.EASY.MAX_PEGS = 1
_C.EASY.N_PEGS = -1

_C.MED = CN()
_C.MED.BASE_RADIUS_LOW_HIGH = [0.0125, 0.025]
_C.MED.BASE_LENGTH_LOW_HIGH = [0.2, 0.3]
_C.MED.PEG_RADIUS_LOW_HIGH = [0.00575, 0.0125]
_C.MED.PEG_LENGTH_LOW_HIGH = [0.1, 0.15]
_C.MED.PEG_ANGLE_LOW_HIGH = [25, 70]
_C.MED.PEG_YAW_LOW_HIGH = [0, 180]
_C.MED.PEG_BASE_HEIGHT_FRAC_LOW_HIGH = [0.25, 0.75]
_C.MED.WITH_BOTTOM = True
_C.MED.BOTTOM_CYLINDER_RADIUS_LOW_HIGH = [0.05, 0.09]
_C.MED.BOTTOM_CYLINDER_LENGTH_LOW_HIGH = [0.002, 0.004] 
_C.MED.BOTTOM_BOX_SIDE_LOW_HIGH = [0.07, 0.1]
_C.MED.BOTTOM_BOX_HEIGHT_LOW_HIGH = [0.001, 0.004]
_C.MED.MAX_PEGS = 2
_C.MED.N_PEGS = 2 

_C.HARD = CN()
_C.HARD.BASE_RADIUS_LOW_HIGH = [0.0125, 0.025]
_C.HARD.BASE_LENGTH_LOW_HIGH = [0.2, 0.3]
_C.HARD.PEG_RADIUS_LOW_HIGH = [0.00575, 0.0125]
_C.HARD.PEG_LENGTH_LOW_HIGH = [0.1, 0.15]
_C.HARD.PEG_ANGLE_LOW_HIGH = [15, 90]
_C.HARD.PEG_YAW_LOW_HIGH = [0, 360]
_C.HARD.PEG_BASE_HEIGHT_FRAC_LOW_HIGH = [0.1, 0.9]
_C.HARD.WITH_BOTTOM = True
_C.HARD.BOTTOM_CYLINDER_RADIUS_LOW_HIGH = [0.05, 0.09]
_C.HARD.BOTTOM_CYLINDER_LENGTH_LOW_HIGH = [0.002, 0.004] 
_C.HARD.BOTTOM_BOX_SIDE_LOW_HIGH = [0.07, 0.1]
_C.HARD.BOTTOM_BOX_HEIGHT_LOW_HIGH = [0.001, 0.004]
_C.HARD.MAX_PEGS = 3
_C.HARD.N_PEGS = -1

def get_syn_rack_default_cfg():
    return _C.clone()
