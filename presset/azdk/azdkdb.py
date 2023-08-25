from os import path
from threading import Thread, Lock
import time
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
import struct
from enum import Enum
import numpy as np
import serial
try: from tqdm import tqdm
except ImportError: tqdm = None
from serial.serialutil import SerialException
from azdk.pgm import Pgm
from azdk.linalg import Vector, Quaternion
from azdk.azdkmisc import AzdkCRC

def jsonfy(obj):
    raise TypeError

def quatntime_from_bytes(d : bytes):
    if len(d) < 24: return (None, None)
    elif len(d) > 24: d = d[:24]
    s = struct.unpack('4iI2H', d)
    q = Quaternion(s[0], s[1], s[2], s[3])
    q.normalize()
    if s[6] == 0: t = None
    else: t = datetime(s[6], 1, 1) + timedelta(s[5]) + timedelta(milliseconds=s[4])
    return (q, t)

def angvelntime_from_bytes(d : bytes):
    if len(d) < 20: return (None, None)
    elif len(d) > 20: d = d[:20]
    s = struct.unpack('3iI2H', d)
    v = Vector(s[0], s[1], s[2])*(2**-30)
    #v.normalize()
    if s[5] == 0: t = None
    else: t = datetime(s[5], 1, 1) + timedelta(s[4]) + timedelta(milliseconds=s[3])
    return (v, t)

def azdk_curtime():
    t = datetime.now()
    ms = t.microsecond / 1000
    t = t.timetuple()
    ms += (t.tm_hour*3600 + t.tm_min*60 + t.tm_sec)*1000
    return (ms, t.tm_yday, t.tm_year)

class AzdkCmd:
    _idcounter = 0
    _codemask = 0x7F

    def __init__(self, _code=0, *, _timeout=0.1) -> None:
        self.code = np.uint8(_code)
        self.params = []
        self.answer = b''
        self._answersize = -1
        self.data = b''
        self._datasize = 0
        self._id = AzdkCmd._idcounter
        AzdkCmd._idcounter += 1
        self.error = 0
        self.updatetime = None
        self.sendtime = None
        self.crc = 0
        self.timeout = _timeout
        self.type = 0

    def __str__(self) -> str:
        code = self.code & self._codemask
        s = f'Id={self._id}, Code={code}'
        np = len(self.params)
        if np > 0:
            s += ' ['
            for kp in range(np):
                if kp: s += ', '
                s += str(self.params[kp])
            s += ']'
        answer = self.answer
        if not isinstance(answer, list | tuple):
            answer = [answer]
        for x in answer:
            if isinstance(x, bytes) and len(x) > 0:
                s += ', answer=' + self.answer.hex(' ').upper()
            else:
                x = str(x)
                if len(x) > 0: s += ', ' + x
        np = len(self.data)
        if np > 0:
            s += f', data of {np} bytes'
        return s

    def clear(self) -> None:
        self._answersize = -1
        self.answer = b''
        self._datasize = 0
        self.data = b''

    def setdata(self, k: int, d: int) -> bool:
        if k < 0 or k >= len(self.params):
            return False
        self.params[k] = type(self.params[k])(d)
        return True

    def isanwerready(self) -> bool:
        return self._answersize == len(self.answer)

    def isready(self) -> bool:
        if self.error > 0: return True
        return len(self.answer) >= self._answersize and len(self.data) >= self._datasize

    def istimeout(self) -> bool:
        tLast = self.updatetime or self.sendtime
        tCurrent = time.perf_counter()
        return tLast and tCurrent > tLast + self.timeout

    def issame(self, cmd, idcounter=False) -> bool:
        ok = cmd and (not idcounter or self._id == cmd._id)
        return ok and (self.code & AzdkCmd._codemask) == (cmd.code & AzdkCmd._codemask)

class AzdkDB:
    CTYPES, CTYPE, DTYPES, DTYPE = 'cmdtypes', 'ctype', 'datatypes', 'datatype'
    CMDS, CMD, PARS, PAR = 'commands', 'cmd', 'params', 'par'
    FLAGS, FLAG, REGS, REG = 'flags', 'flag', 'registers', 'reg'
    ERRS, ERR, MODES, MODE = 'errors', 'err', 'wmodes', 'wmode'
    TYPE, RTYPE, DESCR, NAME, SIZE, VAL, COLOR, CODE, BIT = \
        'type', 'rtype', 'descr', 'name', 'size', 'value', 'color', 'code', 'bit'

    BULK_READ_TYPE = 4

    def __init__(self, xmlpath: str = None):
        self.cmdtypes = {}
        self.datatypes = {}
        self.commands = {}
        self.registers = {}
        self.modes = {}
        self.errors = {}
        self.db = {
            self.CTYPES: self.cmdtypes,
            self.DTYPES: self.datatypes,
            self.CMDS: self.commands,
            self.REGS: self.registers,
            self.MODES: self.modes,
            self.ERRS: self.errors
            }
        if isinstance(xmlpath, str):
            self.loadxml(xmlpath)

    def _loadcommand(self, elem):
        _code = elem.get(self.CODE)
        _type = elem.get(self.TYPE) or '0'
        if not _type or not _code: return
        _code = int(_code)
        _type = int(_type)
        _pars = []
        for par in elem.iterfind(self.PAR):
            _psize = par.get(self.SIZE)
            _ptype = par.get(self.TYPE)
            if not _psize: continue
            try:
                match int(_ptype):
                    case 15: _ptype = float
                    case _: _ptype = int
            except (ValueError, TypeError):
                _ptype = int
            _psize = int(_psize)
            _val = par.get(self.VAL)
            _val = int(_val) if _val else 0
            _flags = []
            for flag in par.iterfind(self.FLAG):
                _bit = flag.get(self.BIT)
                if _bit is None: continue
                _bit = int(_bit) if _bit else 0
                _fval = flag.get(self.VAL)
                _fval = int(_fval) if _fval else 0
                FLAG = {\
                    self.NAME: flag.get(self.NAME), \
                    self.BIT: _bit}
                _flags.append(FLAG)
                _val += _fval << _bit
            param = { \
                self.NAME: par.get(self.NAME), \
                self.SIZE: _psize,
                self.TYPE: _ptype,
                self.VAL: _val}
            if len(_flags) > 0:
                param[self.FLAGS] = _flags
            _pars.append(param)

        _rtype = elem.get(self.RTYPE)
        _rtype = int(_rtype) if _rtype else -1

        _color = elem.get(self.COLOR)

        self.commands[_code] = {self.NAME: elem.get(self.NAME), self.TYPE: _type}
        _descr = elem.get(self.DESCR)
        if _color:
            self.commands[_code][self.COLOR] = _color
        #if RTYPE >= 0:
        self.commands[_code][self.RTYPE] = _rtype
        if _descr:
            self.commands[_code][self.DESCR] = _descr
        if len(_pars) > 0:
            self.commands[_code][self.PARS] = _pars

    def loadxml(self, filepath) -> bool:
        try:
            fl = open(filepath, 'rt', encoding='UTF-8')
        except FileNotFoundError:
            print(f'File {filepath} not found')
            return False
        _d = None
        tree = ET.parse(fl)
        subtree = './/'
        for elem in tree.iterfind(subtree + self.CTYPE):
            TYPE = elem.get(self.TYPE)
            if TYPE is None: continue
            self.cmdtypes[int(TYPE)] = {\
                self.DESCR: elem.get(self.DESCR), \
                self.COLOR: elem.get(self.COLOR)}
        for elem in tree.iterfind(subtree + self.DTYPE):
            TYPE = elem.get(self.TYPE)
            if TYPE is None: continue
            SIZE = elem.get(self.SIZE)
            self.datatypes[int(TYPE)] = {\
                self.NAME: elem.get(self.NAME), \
                self.SIZE: int(SIZE) if SIZE else 0}
        for elem in tree.iterfind(subtree + self.REG):
            CODE = elem.get(self.CODE)
            if CODE is None: continue
            FLAGS = elem.get(self.FLAGS)
            self.registers[int(CODE)] = {\
                self.NAME: elem.get(self.NAME), \
                self.DESCR: elem.get(self.DESCR), \
                self.FLAGS: int(FLAGS) if FLAGS else 0}
        for elem in tree.iterfind(subtree + self.CMD):
            self._loadcommand(elem)
        for elem in tree.iterfind(subtree + self.ERR):
            CODE = elem.get(self.CODE)
            if CODE is None: continue
            self.errors[int(CODE)] = {\
                self.NAME: elem.get(self.NAME), \
                self.DESCR: elem.get(self.DESCR)}
        for elem in tree.iterfind(subtree + self.MODE):
            TYPE = elem.get(self.TYPE)
            CODE = elem.get(self.CODE)
            if TYPE is None or CODE is None: continue
            self.modes[(int(TYPE), int(CODE))] = {\
                self.NAME: elem.get(self.NAME), \
                self.DESCR: elem.get(self.DESCR)}
        return True

    def dtype(self, _dtype: int):
        try: return self.datatypes[_dtype]
        except KeyError: return None

    def showcommands(self, jsonlike=False, verbose=False):
        if jsonlike:
            print(json.dumps(self.commands, indent=2, ensure_ascii=False))
        else:
            for k, v in self.commands.items():
                s = f'{k:02X}h: {v[self.NAME]}, type={v[self.TYPE]}'
                if verbose:
                    if self.RTYPE in v:
                        s += f', rtype={v[self.RTYPE]}'
                    if self.DESCR in v:
                        s += f', descr={v[self.DESCR]}'
                    if self.PARS in v:
                        for p in v[self.PARS]:
                            s += f'\n - {p[self.NAME]}, size={p[self.SIZE]}'
                print(s)

    def _showmodes(self, jsonlike=False):
        if jsonlike:
            _dict = {}
            for k, v in self.modes.items():
                _dict[f'{k[0], k[1]}'] = v
            return json.dumps(_dict, indent=2, ensure_ascii=False)
        else:
            for k, v in self.modes.items():
                print(f'{k}h: {v["name"]} ({v["descr"]})')

    def tojson(self, filepath: str):
        try: fl = open(filepath, 'wt', encoding='UTF-8')
        except IOError:
            print('Path does not exist or no write access')
            return
        MODES = {}
        for k1, k2 in self.modes:
            MODES[f'{k1}.{k2:02d}'] = self.modes[(k1, k2)]
        self.db[self.MODES] = MODES
        json.dump(self.db, fl, indent=2, ensure_ascii=False, default=jsonfy)

    def info(self) -> str:
        s = 'AZDK database'
        s += f': {len(self.commands)} commands'
        s += f', {len(self.datatypes)} data types'
        s += f', {len(self.registers)} registers'
        s += f', {len(self.modes)} modes'
        s += f', {len(self.errors)} error types'
        return s

    def _createcmd(self, code, vcmd, params, timeout):
        cmd = AzdkCmd()
        cmd.code = code
        if timeout: cmd.timeout = timeout
        rtype = vcmd[self.RTYPE] if self.RTYPE in vcmd else None
        if rtype in self.datatypes:
            sz = self.datatypes[rtype][self.SIZE]
            if sz: cmd._answersize = sz
        if self.PARS in vcmd:
            for pk, p in enumerate(vcmd[self.PARS]):
                sz = p[self.SIZE]
                pval = 0
                if hasattr(params, '__iter__') and len(params) > pk:
                    pval = int(params[pk])
                if sz == 1:
                    cmd.params.append(np.uint8(pval))
                elif sz == 2:
                    cmd.params.append(np.uint16(pval))
                elif sz == 4:
                    cmd.params.append(np.uint32(pval))
        cmd.type = vcmd[self.TYPE]
        return cmd

    def createcmd(self, codename : int, params: list = None, *, timeout=0.1) -> AzdkCmd:
        if params is not None and not hasattr(params, '__iter__'):
            params = [params]
        if isinstance(codename, str):
            for code, cmd in self.commands.items():
                if cmd[self.NAME] == codename:
                    return self._createcmd(code, cmd, params, timeout)
        elif isinstance(codename, int):
            if codename in self.commands:
                return self._createcmd(codename, self.commands[codename], params, timeout)
        return None

    def answer(self, cmd : AzdkCmd):
        return self.interpret(cmd.code, cmd.answer)

    def interpret(self, code : int, answer : bytes):
        code = code & AzdkCmd._codemask
        if not code in self.commands:
            return answer
        rtype = self.commands[code][self.RTYPE]
        match rtype:
            case  4: return struct.unpack('i', answer)[0]
            case  3: return struct.unpack('I', answer)[0]
            case  2: return struct.unpack('H', answer)[0]
            case  1: return struct.unpack('B', answer)[0]
            case 10: return quatntime_from_bytes(answer)
            case 16: return angvelntime_from_bytes(answer)
            case 12: return np.frombuffer(answer, np.int32)
            case 11:
                if len(answer) == 36:
                    return struct.unpack('I2HI2HI2H3i', answer)
                elif len(answer) == 24:
                    return struct.unpack('I2HI2HI2H', answer)
                elif len(answer) == 16:
                    return struct.unpack('I2HI2H', answer)
            case  7:
                d = np.frombuffer(answer, np.int32)
                if len(d) == 4: return Quaternion(d*(2**-30))
            case 6:
                d = np.frombuffer(answer, np.int32)
                if len(d) == 3: return Vector(d*(2**-30))
        return answer

    def answer_str(self, cmd: AzdkCmd) -> str:
        answer = self.interpret(cmd.code, cmd.answer)
        if isinstance(answer, tuple) or isinstance(answer, np.ndarray):
            return ', '.join(str(x) for x in answer)
        if isinstance(answer, bytes):
            return cmd.answer.hex(' ').upper()
        return str(answer)

    def cmdinfo(self, cmd: AzdkCmd) -> str:
        code = cmd.code & 0x7F
        if not code in self.commands:
            return None
        s = f'CMD({code}, id={cmd._id}), '
        s += self.commands[code][self.NAME]

        if len(cmd.params) > 0:
            s += ', p=['
            for p in cmd.params: s += str(p)
            s += ']'

        answ_str = self.answer_str(cmd)
        if isinstance(answ_str, str) and len(answ_str) > 0:
            s += ': ' + answ_str
        return s

    def cmdname(self, cmd: AzdkCmd) -> str:
        if not cmd.code in self.commands:
            return None
        return self.commands[cmd.code][self.NAME]

    def cmd(self, _code: int) -> AzdkCmd:
        return self.commands[_code]

    def findcmd(self, _name: str):
        for _, v in self.commands.items():
            if v[self.NAME] == _name:
                return v
        return None

    def error(self, _err: int) -> str:
        if _err in self.errors:
            return self.errors[_err]
        else:
            return ''

class WState:
    class signal(Enum):
        SSF_SHTR_CLOSE = 1          # флаг закрытия шторки
        SSF_AM_MODE = 2             # включение режима поддержания ориентации
        SSF_CRIT_ERR_INT = 4        # критическая ошибка
        SSF_HIGH_DARK = 8           # слишком высокий темновой ток
        SSF_HIGH_BKGR = 16          # слишком высокая засветка
        SSF_NEED_DARK_CAL = 32      # необходимость калибровки темнового тока
        SSF_ABORT = 64              # отмена текущего режима работы
        SSF_PELTIER = 128           # флаг включение пельтье
        SSF_CMD_EXEC = 256          # осуществляется исполнение команды
        SSF_FRAME_READY = 512       # готовность нового кадра для обработки
        SSF_QUAT_RECEIVED = 1024    # успешное измерение кватерниона
        SSF_FRAME_SENT = 2048       # флаг отсылки кадра
        SSF_SPEC_DATA_RX = 4096     # факт приема данных в спецрежиме
        SSF_SPEC_DATA_TX = 8192     # факт передачи данных в спецрежиме
        SSF_DATA_STORED = 16384     # факт наличия данных во Flash
        SSF_DB_COPIED = 32768       # факт наличия копий БД во Flash

    def __init__(self) -> None:
        self.wmode = 0
        self.rmode = 0
        self.specmode = False
        self.progress = 0.0
        self.werr = 0
        self.cmderr = 0
        self.cmdlast = 0
        self.temp = 0
        self.sflags = 0

        self.readstage = 0
        self.calerr = 0
        self.cmosflags = 0

        self.backgr = 0
        self.dark = 0

    def __str__(self) -> str:
        s = 'Azdk state: '
        s += f'mode={self.wmode}, rd_mode={self.rmode}, sp_flag={self.specmode}, progress={self.progress}'
        s += f', werr={self.werr}, cerr={self.cmderr}, last_cmd={self.cmdlast}'
        s += f', T={self.temp}, flags={self.sflags}, star_cnt={self.calerr}'
        s += f', backlight={self.backgr}, dark={self.dark}'
        return s

    def from_buffer(self, d: bytes):
        if len(d) != 16: return
        self.wmode = d[0] & 0x0F
        self.rmode = (d[0] >> 4) & 0x07
        self.specmode = d[0] > 127
        self.progress = float(d[1]) / 2.25
        self.werr = d[2]
        self.cmderr = d[3]
        self.cmdlast = d[4]
        self.temp = d[5] * 0.5 - 40.0
        self.sflags = np.uint16(d[6] + d[7]*256)
        self.readstage = d[8]
        self.calerr = d[9]
        self.cmosflags = np.uint16(d[10] + d[11]*256)
        self.backgr = np.uint16(d[12] + d[13]*256)
        self.dark = np.uint16(d[14] + d[15]*256)

    @classmethod
    def from_bytes(cls, d: bytes):
        if len(d) != 16: return None
        ws = cls()
        ws.from_buffer(d)
        return ws

class WMode(Enum):
    Unknown = 0
    Init = 1
    Idle = 2
    Debug = 3
    Autonomous = 4
    DarkCalibration = 5
    MemsCalibration = 6
    WriteToFlash = 7
    SoftwareUpdate = 8
    SoftwareRecover = 9
    FreeFrameRead = 10
    CmdSync = 11
    Test = 12
    FreeFrameReadCont = 13
    FreeFrameReadAvg = 14
    Loader = 15
    FFRMeanAndStd = 16

class AzdkPort:
    def __init__(self, portname, baudrate) -> None:
        self.port = serial.Serial(portname, baudrate) if portname else None
        self.cmd = None
        self.buffer = b''
        self.cmdidcounter = False
        self.echo = False

    def open(self, portname=None, baudrate=None) -> bool:
        if not self.port:
            self.port = serial.Serial()
        elif self.port.isOpen():
            self.port.close()
        if portname: self.port.name = portname
        if baudrate: self.port.baudrate = baudrate
        try:
            self.port.open()
            return True
        except SerialException:
            return False

    def close(self):
        if self.port:
            self.port.close()

    def cmdtobytes(self, cmd: AzdkCmd):
        raise NotImplementedError()

    def readbuffer(self, expectedCmd: AzdkCmd = None) -> AzdkCmd:
        raise NotImplementedError()

    def isbusy(self) -> bool:
        return self.cmd is not None

class AzdkRS485(AzdkPort):
    _address = b'\xA0'

    def __init__(self, portname=None, baudrate=115200, parity='N', stopbits=1, databits=8) -> None:
        super().__init__(portname, baudrate)
        if databits: self.port.bytesize = databits
        if stopbits: self.port.stopbits = stopbits
        if parity: self.port.parity = parity

    def sendarray(self, d: bytes):
        if self.port and self.port.isOpen():
            #self.cmd = None
            self.port.write(d)
            self.port.flush()
            if self.echo:
                print('TX: ' + d.hex(' ').upper())

    def cmdtobytes(self, cmd: AzdkCmd):
        d = b''
        d += self._address
        d += np.uint8(cmd.code)
        if self.cmdidcounter:
            d += np.uint8(cmd._id)
        dd = b''
        for x in cmd.params:
            dd += x.tobytes()
        d += np.uint8(len(dd))
        d += dd
        d += AzdkCRC.crc8(d)
        return d

    def _readanswer(self) -> bool:
        sz = self.cmd._answersize
        self.cmd.answer = self.buffer[:sz]
        self.cmd.updatetime = time.perf_counter()
        self.cmd.crc = AzdkCRC.crc8(self.cmd.answer, self.cmd.crc)
        crc = self.buffer[sz]
        self.buffer = self.buffer[sz+1:]
        if self.cmd.crc != crc:
            return False
        if self.cmd.type == AzdkDB.BULK_READ_TYPE and len(self.cmd.answer) >= 4:
            self.cmd._datasize = int.from_bytes(self.cmd.answer[:4], 'little')
        return True

    def _readdata(self):
        ks = self.buffer.find(self._address)
        szBuf = len(self.buffer)
        # check header package size
        kk = ks + len(self._address)
        if ks < 0 or szBuf < kk + 4: return
        pkgSz = self.buffer[kk + 3]
        kn = kk + 5 + pkgSz
        # check full package size
        if szBuf < kn: return
        #TODO check package number
        #pkgNum = self.buffer[kk + 1] + self.buffer[kk + 2]*256
        # check command code and skip package
        if (self.cmd.code & AzdkCmd._codemask) != self.buffer[kk] or pkgSz < 1:
            self.buffer = self.buffer[kn:]
            return
        if self.buffer[kn-1] != AzdkCRC.crc8(self.buffer[ks:kn-1]):
            self.cmd.error = 255
        self.cmd.updatetime = time.perf_counter()
        self.cmd.data += self.buffer[kk+4:kn-1]
        self.buffer = self.buffer[kn:]

    def readbuffer(self, expectedCmd: AzdkCmd = None) -> AzdkCmd:
        self.buffer += self.port.read_all()
        if self.cmd is None:
            k = self.buffer.find(self._address)
            kp = k + (5 if self.cmdidcounter else 4)
            if k < 0 or len(self.buffer) < kp: return
            kk = k + len(self._address) + 1
            self.cmd = AzdkCmd()
            self.cmd.code = self.buffer[kk - 1]
            if self.cmdidcounter:
                self.cmd._id = self.buffer[kk]
                kk += 1
            if self.cmd.issame(expectedCmd, self.cmdidcounter):
                self.cmd = expectedCmd
            sz = self.buffer[kk]
            self.cmd.crc = AzdkCRC.crc8(self.buffer[k:kp-1])
            self.buffer = self.buffer[kp-1:]
            if sz > 0xF0 or sz == 0:
                self.cmd._answersize = 0
                self.cmd.error = sz
                crc = self.buffer[0]
                self.buffer = self.buffer[1:]
                cmd = self.cmd
                self.cmd = None
                return cmd if cmd.crc == crc else None
            self.cmd._answersize = sz
        else:
            sz = self.cmd._answersize
        if len(self.cmd.answer) == sz:
            self._readdata()
        elif len(self.buffer) > sz and len(self.cmd.answer) == 0:
            if not self._readanswer():
                self.cmd = None
                return None
        if self.cmd and self.cmd.isready():
            cmd = self.cmd
            self.cmd = None
            if self.echo:
                print('cmd ' + str(cmd) + ' received')
            return cmd

class AzdkConnect(Thread):
    _maxqueuesize = 16
    ParityEven = 'E'
    ParityOdd = 'O'
    ParityNone = 'N'

    def __init__(self, portname, baudrate, *, porttype=AzdkRS485, parity='E', stopbits=1, databits=8, idcounter=False, verbose=False) -> None:
        super().__init__()
        self.cmdqueue = []
        self.cmdsent = None
        self.cmdready = []
        self.portname = portname
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.databits = databits
        self._mutex = Lock()
        self._running = False
        self._stopped = False
        self._porttype = porttype
        self._idcmdcounter = idcounter
        self.verbose = verbose

    def _sendcmd(self, port: AzdkPort):
        if port.isbusy(): return False
        if self.cmdsent: return False
        if len(self.cmdqueue) == 0: return False
        if not self._mutex.acquire(False): return False
        self.cmdsent = self.cmdqueue[0]
        del self.cmdqueue[0]
        d = port.cmdtobytes(self.cmdsent)
        if self.verbose:
            print(f'cmd {self.cmdsent} sent')
        self._mutex.release()
        self.cmdsent.sendtime = time.perf_counter()
        self.cmdsent.updatetime = None
        port.sendarray(d)
        return True

    def _check_is_in_queue(self, cmd: AzdkCmd):
        self._mutex.acquire()
        ok = cmd in self.cmdqueue
        self._mutex.release()
        return ok

    def _check_is_sent(self, cmd: AzdkCmd):
        self._mutex.acquire()
        ok = self.cmdsent and (cmd == self.cmdsent)
        self._mutex.release()
        return ok

    def _check_cmd_isready(self, cmd: AzdkCmd):
        self._mutex.acquire()
        ok = (cmd in self.cmdready)
        self._mutex.release()
        return ok

    def _enqueueready(self, cmd: AzdkCmd):
        self._mutex.acquire()
        #replace notifications
        for kr, cmdr in enumerate(self.cmdready):
            if cmd.code == cmdr.code and cmd._id == cmdr._id:
                self.cmdready[kr] = cmd
                break
        else:
            self.cmdready.append(cmd)
        self._mutex.release()

    def run(self):
        self._running = True
        if self.verbose:
            print(f'Starting port {self._porttype} thread')
        if self._porttype == AzdkRS485:
            _port = AzdkRS485(self.portname, self.baudrate, self.parity, self.stopbits, self.databits)
            _port.cmdidcounter = self._idcmdcounter
        elif self._porttype is None:
            self._running = False
            self._stopped = True
            return False
        else:
            _port = self._porttype(self.portname, self.baudrate)
        if not _port.open(): return False

        _port.echo = self.verbose

        while self._running:
            if self._sendcmd(_port):
                time.sleep(0.02)
            cmd = _port.readbuffer(self.cmdsent)
            if self.cmdsent:
                if self.cmdsent.issame(cmd, self._idcmdcounter):
                    self.cmdsent.answer = cmd.answer
                    self.cmdsent.data = cmd.data
                    cmd = self.cmdsent
                    self.cmdsent = None
                elif self.cmdsent.istimeout():
                    if self.verbose:
                        print('cmd ' + str(self.cmdsent) + f' timeout ({self.cmdsent.timeout} sec)')
                    self.cmdsent = None
                    _port.cmd = None
            if cmd:
                self._enqueueready(cmd)
                cmd = None
            time.sleep(0.01)
        _port.close()
        self._stopped = True
        if self.verbose:
            print(f'Port {self._porttype} has been stopped')
        return True

    def stop(self):
        self._running = False
        self.join()

    def enqueuecmd(self, cmd: AzdkCmd):
        if not self.is_alive():
            return False
        if len(self.cmdqueue) < self._maxqueuesize and self._mutex.acquire():
            cmd.clear()
            self.cmdqueue.append(cmd)
            self._mutex.release()
            time.sleep(0.01)
            return True
        return False

    def waitforanswer(self, cmd: AzdkCmd):
        if not self.is_alive():
            return False
        while self._check_is_in_queue(cmd):
            time.sleep(0.01)
        ok = self._check_cmd_isready(cmd)
        if not self._check_is_sent(cmd) and not ok:
            return False
        while not ok:
            time.sleep(0.01)
            if cmd.istimeout():
                return False
            ok = self._check_cmd_isready(cmd)
        self._mutex.acquire()
        self.cmdready.remove(cmd)
        self._mutex.release()
        return True

    def get_notification(self, code: int = None) -> AzdkCmd | None:
        cmd = None
        self._mutex.acquire()
        for k, xcmd in enumerate(self.cmdready):
            if code is None or xcmd.code == code:
                cmd = self.cmdready[k]
                del self.cmdready[k]
                break
        self._mutex.release()
        return cmd

    def sendcmd(self, cmd: AzdkCmd, db: AzdkDB = None) -> bool:
        if not self.enqueuecmd(cmd):
            return False
        if not self.waitforanswer(cmd):
            return False
        if cmd.error > 0:
            if db and self.verbose:
                print(f'Error on cmd {db.cmdname(cmd)}: {db.error(cmd.error)}')
            return False
        return True

    def getframe(self, db: AzdkDB = None, *, showprogress=True, timeout=1, subframe: list = None) -> Pgm:
        if subframe:
            cmdGetFrame = db.createcmd(24, subframe, timeout=timeout)
        else:
            cmdGetFrame = db.createcmd(23, timeout=timeout)

        if not self.enqueuecmd(cmdGetFrame):
            return None
        while not cmdGetFrame.isanwerready():
            time.sleep(0.1)
            if cmdGetFrame.istimeout():
                return None

        hdr = np.frombuffer(cmdGetFrame.answer[4:], np.uint16)
        nsec = int(hdr[0]) * int(hdr[1]) * 12 / self.baudrate

        pbar = None
        if showprogress and tqdm and nsec > 3.0:
            pbar = tqdm(desc='Obtaining frame', bar_format='{l_bar}{bar}', total=cmdGetFrame._datasize)

        while not cmdGetFrame.isready():
            time.sleep(0.1)
            if pbar:
                pbar.n = len(cmdGetFrame.data)
                pbar.update(0)
            if cmdGetFrame.istimeout():
                return None

        if pbar: pbar.close()

        data = np.frombuffer(cmdGetFrame.data, np.uint16)
        pgm = Pgm(hdr[0], hdr[1], dtype=np.uint16, maxval=max(data), _d=data)
        if len(hdr) > 2:
            pgm._pars['id'] = hdr[2] + hdr[3]*65536
        if len(hdr) > 4:
            pgm._pars['dark'] = hdr[4]
        if len(hdr) > 5:
            pgm._pars['exp'] = hdr[5]
        if len(hdr) > 9:
            mses = hdr[6] + hdr[7]*65536
            pgm._pars['time'] = datetime(hdr[9], 1, 1) + timedelta(int(hdr[8]), seconds=mses*0.001)
        if len(hdr) > 15:
            x = hdr[10] + hdr[11]*65536
            y = hdr[12] + hdr[13]*65536
            z = hdr[14] + hdr[15]*65536
            pgm._pars['vel'] = Vector(x, y, z) * (2**-30 * 180/np.pi)
        return pgm

    def clearready(self):
        self._mutex.acquire()
        self.cmdready.clear()
        self._mutex.release()

def azdkdbtest1(xmlpath: str):
    db = AzdkDB(xmlpath)

    conn = AzdkConnect('COM3', 500000, idcounter=True)
    conn.start()

    while conn.is_alive():
        qcmd_notif = conn.get_notification()
        if qcmd_notif:
            print(db.cmdinfo(qcmd_notif))
        #cmd = db.createcmd(16)
        #cmd.timeout = 1.0
        #if conn.enqueuecmd(cmd):
        #    if conn.waitforanswer(cmd):
        #        print(db.cmdinfo(cmd))
        time.sleep(0.05)

def azdkdbtest2(xmlpath: str):
    db = AzdkDB(xmlpath)
    print(db.info())
    cmd = db.createcmd(16)
    cmd_d = AzdkRS485().cmdtobytes(cmd)
    print(cmd, 'cmd_data=' + cmd_d.hex(' ').upper(), sep=', ')

def azdkdbtest3(xmlpath: str):
    wdir, filename = path.split(xmlpath)
    db = AzdkDB(xmlpath)
    print(db.info())
    db.showcommands(verbose=True)
    db._showmodes()
    db.tojson(wdir + filename + '.json')

if __name__ == "__main__":
    azdkdbtest1('G:/programming/funktest/AZDKHost.xml')
