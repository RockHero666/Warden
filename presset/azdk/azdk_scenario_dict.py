import os
from enum import Enum
import azdk.azdksocket as azs
from azdk.azdkdb import AzdkDB

class AzdkCommands(Enum):
    RESTART = 1
    RESET_SETTINGS = 2
    SET_DATETIME = 3
    SET_REGISTER = 4
    SET_CTRL_FLAGS = 5
    SET_SAT_VEL = 6
    SET_ANGVEL = 7
    SET_EXP_TIME = 8
    SET_FRAME_RECT = 9
    SET_SENSOR_FLAGS = 10
    SET_SETTINGS = 11
    SET_SAT_REF = 13
    TOGGLE_SHUTTER = 14
    TOGGLE_PELTIER = 15

    GET_STATE = 16
    GET_REGISTER = 17
    GET_ANGVEL = 18
    GET_ANGVEL_MODEL = 19
    GET_ATTITUDE = 20
    GET_DATETIME = 21
    GET_PHC_LIST = 22
    GET_FRAME = 23
    GET_SUBFRAME = 24
    #GET_WINDOWS = 25
    GET_STATISTICS = 26
    GET_SETTINGS = 27
    #GET_TARGET = 28
    #GET_OBJECT_LIST = 29
    SET_RS485_SPEED = 30
    SET_CAN_SPEED = 31

    SET_IDLE_MODE = 48
    SET_AUTO_MODE = 49
    #SET_SYNC_MODE = 50
    #SET_DARK_CALIBRATION_MODE = 51
    SET_AVC_CALIBRATION_MODE = 52
    SET_RAW_FRAME_MODE = 53
    SET_FRAME_AVG_MODE = 57

    WRITE_ROM_DATA = 64
    SAVE_SETTINGS = 65
    READ_RAM_DATA = 68
    UPLOAD_RAM_DATA = 69
    GET_FW_VERSION = 70
    UPDATE_FW = 71
    SET_DEVICE_ID = 72

    #UPLOAD_FW_PART = 73
    BOOTLOAD_MODE = 75
    SETUP_NOTIFICATIONS = 76

    #NOTIF_RESTART = 112
    #NOTIF_STATE = 113
    #NOTIF_QUAT = 114
    #NOTIF_ANGVEL = 115
    #NOTIF_TARGET = 117
    #NOTIF_PHCS = 118

    @classmethod
    def getname(cls, code: int):
        for _name, _value in cls.__members__.items():
            if _value.value == code:
                return _name
        return None

    @property
    def code(self) -> int:
        return super().value

    @property
    def descr(self, db : AzdkDB) -> str:
        return db.commands[self.value]['name']

class ScenarioCommands(Enum):
    EXEC_FUNC_TEST = (1, 'Проведение функционального теста')

    @classmethod
    def getname(cls, code: int):
        for _name, _value in cls.__members__.items():
            if _value.value[0] == code:
                return _name
        return None

    @property
    def code(self) -> int:
        return super().value[0]

    @property
    def descr(self) -> str:
        return super().value[1]

class AzdkCmdStruct:
    def __init__(self, code, name, descr):
        self.name = name
        self.code = code
        self.descr = descr

def _create_dict(db : AzdkDB = None):
    d = dict()
    if db is None:
        db = AzdkDB('d:/Users/Simae/Work/2019/PDStand/Win32/Release/AZDKHost.xml')
    for name, cmd in azs.AzdkServerCommands.__members__.items():
        if not hasattr(cmd.value, '__getitem__'): continue
        name = 'azs_' + name.lower()
        d[name] = cmd
    for name, cmd in azs.PDSServerCommands.__members__.items():
        name = name.lower()
        if not hasattr(cmd.value, '__getitem__'): continue
        name = 'pds_' + name.lower()
        d[name] = cmd
    for name, code in AzdkCommands.__members__.items():
        _name =  'azdk_' + name.lower()
        try: d[_name] = AzdkCmdStruct(code.value, name, db.commands[code.value]['name'])
        except KeyError: pass
    for name, cmd in ScenarioCommands.__members__.items():
        d['sc_' + name.lower()] = cmd
    return d

class ScenarioCmd:
    _dict = _create_dict()
    _azdkdb = AzdkDB()

    def __init__(self, cmd : ScenarioCommands | azs.PDSServerCommands | azs.AzdkServerCommands | AzdkCmdStruct,
                 params : list = None, timeout = 0.1, iscritical = False, executor : azs.AzdkSocket = None):
        self.cmd = cmd
        self.executor = executor
        self.iscritical = iscritical
        self.timeout = timeout
        self.params = params
        self.cmd_ex = self._createcmd()

    @property
    def code(self):
        return self.cmd.code

    @property
    def name(self):
        return self.cmd.name

    @property
    def descr(self):
        return self.cmd.descr

    def _createcmd(self):
        if isinstance(self.cmd, ScenarioCommands):
            return self.cmd.code
        elif isinstance(self.cmd, azs.PDSServerCommands):
            return azs.PDSServerCmd(self.cmd, self.params, None, self.timeout)
        elif isinstance(self.cmd, azs.AzdkServerCommands):
            return azs.AzdkServerCmd(self.cmd, self.params, None, self.timeout)
        elif isinstance(self.cmd, AzdkCmdStruct):
            return self._azdkdb.createcmd(self.cmd.code, self.params, timeout=self.timeout)
        return None

    def exec(self):
        if self.executor is None: return False
        if self.cmd_ex is None: return False
        timeout = self.timeout*1.5
        if isinstance(self.cmd, AzdkCmdStruct):
            return azs.call_azdk_cmd(self.executor, self.cmd_ex, timeout)
        if isinstance(self.executor, azs.AzdkSocket):
            return self.executor.execute(self.cmd_ex, timeout)
        return self.executor(self.cmd_ex)

def scenario_dict_test():
    _dict = _create_dict()
    for cmd, info in _dict.items():
        print(f'{cmd}: {info.code}, {info.name},  {info.descr} ({type(info)})')

if __name__ == "__main__":
    scenario_dict_test()
    os.system('pause')
