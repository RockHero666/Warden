from enum import Enum

class AzdkCommands(Enum):
    RESTART                  = 1           
    RESET                    = 2          
    DATE                     = 3            
    REGISTR                  = 4 
    SET_FLAGS                = 5         
    CHANGE_FLAGS             = 32        
    ANGL_SPEED_KA            = 6    
    ANGL_SPEED               = 7       
    DURATION_OF_ACCUMULATION = 8   
    SCREEN_GEOMETRY          = 9   
    FPY_FLAGS                = 10        
    SET_PARAMS               = 11
    SET_SK_KA                = 13
    CURTAIN                  = 14
    PELTIER                  = 15
    STATE                    = 16
    REED_AZDK_REGISTR        = 17
    REED_ANGLE_SPEED         = 18
    REED_ANGLE_SPEED_MODEL   = 19
    REED_LAST_QUAT           = 20
    REED_DATE_TIME           = 21
    REED_LIST_FOTO           = 22
    REED_FRAME               = 23
    REED_SUBSCREEN           = 24
    REED_WINDOWS             = 25
    STATISTICS               = 26
    REED_PARAMS              = 27
    REED_FOCUS_OBJECT        = 28
    REED_LIST_NON_STARS      = 29
    SET_SPEED_RS485          = 30
    SET_SPEED_CAN            = 31
    STANDBY_MODE             = 48
    AUTO_MODE                = 49
    COMMAND_MODE             = 50
    CALIB_DARK_CURRENT_MOD   = 51
    CALIB_MEMS_GIRO_MOD      = 52
    REED_RAW_FRAME_MODE      = 53
    FRAME_AVERAGE_MODE       = 57
    SAVE_ALL_DATA_FLASH      = 64
    SAVE_PROPERTIES_FLASH    = 67
    REED_DATA                = 68
    SAVE_DATA                = 69
    READ_FW_VERSION          = 70
    UPDATE_SOFT              = 71
    SET_NUMBER_AZDK          = 72
    OVERWRITING_SOFT         = 73
    RETURN_TO_LOADER         = 75
    PROPERTIES_NOTIFICATION  = 76

    
    @classmethod
    def getname(cls, code: int):
        for _name, _value in cls.__members__.items():
            if _value.value == code:
                return _name
        return None

    @classmethod
    def findname(cls, name: str):
        return cls.__members__[name].value