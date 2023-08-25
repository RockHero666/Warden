import os
import time
import datetime as dt
import numpy as np
from tqdm import tqdm
from azdk.azdkdb import AzdkDB, WState, WMode, AzdkCmd
from azdk.azdksocket import AzdkSocket, PDSServerCmd, PDSServerCommands, AzdkServerCommands, call_azdk_cmd
from azdk.linalg import Quaternion, Vector
from azdk.funtest_analyzer import FunTestAnalyzer
#from azdk.funtest_analyzer import azdk_fun_test_results_analyzer
from azdk.pdfcreator import FunTestPdfCreator
from azdk.utils import AzdkLogger
from azdk.telebot import TeleBot

class AzdkFlags:
    def __init__(self, flags=0):
        self.SCF_AUTONOMOUS = flags & (1 << 0)      # вкл. перехода в режим автономной работы после инициализации АЗДК
        self.SCF_HYRO_READ = flags & (1 << 1)       # вкл. использования гироскопа
        self.SCF_RESERVE1 = flags & (1 << 2)        #
        self.SCF_RESET_STATS = flags & (1 << 3)     # вкл. обнуления статистики при рестарте
        self.SCF_WATCHDOG_EN = flags & (1 << 4)     # вкл. сторожевого таймера
        self.SCF_BIN_MODE_DIS = flags & (1 << 5)    # вкл. режима бинирования
        self.SCF_XSPEC_MODE = flags & (1 << 6)      # вкл. спец режима работы с АЗДК
        self.SCF_SHUTTER_POL = flags & (1 << 7)     # задание полярности переключения шторки
        self.CCR_CALC_SM_ABER = flags & (1 << 8)    # вкл. учета аберрации света за счет движения КА
        self.CCR_REFINE_ATT = flags & (1 << 9)      # вкл. уточнения ориентации
        self.CCR_SAT_RF = flags & (1 << 10)         # вкл. перевода координат в СК КА вместо СК АЗДК
        self.CCR_II_REDUCTION = flags & (1 << 11)   # вкл. фотометрической нормализации кадра в режиме РНО
        self.CCR_USE_WINDS = flags & (1 << 12)      # вкл. использования оконного считывания
        self.CCR_DISABLE_AM = flags & (1 << 13)     # выкл. режима поддержания ориентации
        self.CCR_AMPLIFY = flags & (1 << 14)        # вкл. усиления сигнала
        self.CCR_NO_BLUR_STARS = flags & (1 << 15)  # вкл. функции адаптивного изменения времени накопления в зависимости от смаза

    def value(self):
        return sum(vars(self))

    def __str__(self):
        s = [a for a, v in vars(self).items() if v != 0]
        return ', '.join(s)

class AzdkSettings:
    def __init__(self, *, logger = AzdkLogger(timestamp_fl=False)):
        self.frequency = 0.0
        self.exptime = 0
        self.azdknum = 0
        self.azdkser = 0
        self.fw_version = '1.0.0000'
        self.logger = logger
        self.devname = ''
        self.flags = AzdkFlags()
        self.timeout_default = 1.0

    def getinfo(self, azs : AzdkSocket, db : AzdkDB, verbose=True):
        self.logger.log('Get device info')
        cmd = db.createcmd(17, 22, timeout=self.timeout_default*0.5)
        if call_azdk_cmd(azs, cmd, self.timeout_default):
            self.azdknum = cmd.answer[1]*100 + cmd.answer[0]
            self.azdkser = cmd.answer[3]*100 + cmd.answer[2]
            self.devname = f'АЗДК-{cmd.answer[3]}.{cmd.answer[2]} №{cmd.answer[1]}-{cmd.answer[0]:02d}'
            if verbose: self.logger.log(f"  device id: '{self.devname}'")
        cmd = db.createcmd(70, timeout=self.timeout_default*0.5)
        if call_azdk_cmd(azs, cmd, self.timeout_default):
            self.fw_version = f'{cmd.answer[3]}.{cmd.answer[2]}.{cmd.answer[1]:02X}{cmd.answer[0]:02X}'
            if verbose: self.logger.log(f"  firmware version: '{self.fw_version}'")
        cmd = db.createcmd(17, 16, timeout=self.timeout_default*0.5)
        if call_azdk_cmd(azs, cmd, self.timeout_default):
            self.exptime = cmd.answer[1]*100 + cmd.answer[0]
            if verbose: self.logger.log(f'  exp.time: {self.exptime}')
        cmd = db.createcmd(17, 15, timeout=self.timeout_default*0.5)
        if call_azdk_cmd(azs, cmd, self.timeout_default):
            self.frequency = 1000/(cmd.answer[1]*256 + cmd.answer[0])
            if verbose: self.logger.log(f'  frequency: {self.frequency}')
        cmd = db.createcmd(32, 0, timeout=self.timeout_default*0.5)
        if call_azdk_cmd(azs, cmd, self.timeout_default):
            self.flags = AzdkFlags(cmd.answer[0] + cmd.answer[1]*256)
            if verbose: self.logger.log(f'  flags: {self.flags}')

class FunTestExecutor:
    statsRequestInterval = 5.0

    def __init__(self, wdir : str, *, azdkdbpath='AZDKHost.xml', save_figures=False):
        self.tracks = []
        logfile = wdir + '/logs//azdk_fun_test_' + dt.datetime.now().strftime(r'%Y.%M.%d_%H.%m.%S') + '.log'
        self.logger = AzdkLogger(logfile)
        self.pdf = FunTestPdfCreator(logger=self.logger)
        self.device = AzdkSettings()
        self.wdir = wdir
        self.azdkdb = AzdkDB(azdkdbpath)
        self.pds = AzdkSocket()
        self.azs = AzdkSocket()
        self.save_figures = save_figures
        self.duration = 0.0
        self.count = 0
        self.analyzer = FunTestAnalyzer(True, self.logger)
        self.analyzer.waitUntilStart()
        self.stats = []
        self.timeout_default = 1.0

    def log(self, msg : str):
        print(msg)
        if self.logger: self.logger.log(msg)

    def openAzdkDatabase(self, path : str):
        self.azdkdb.loadxml(path)

    def restartPdfDaemon(self, author='Максим Тучин', pdf_name=None, verbose=True,
                         date_beg=dt.date.today(), date_end=dt.date.today(),
                         comm_speed=500, save_to_file=True):
        if self.pdf.is_alive():
            self.pdf.stop()
        if pdf_name is None:
            pdf_name = 'ФИ ' + self.device.devname
        t = dt.datetime.now()
        self.pdf.downscale_images = False
        self.pdf.autosave_path = self.wdir + pdf_name + f'.{t:%H_%M}.pdf' if save_to_file else None
        self.pdf.author = author
        self.pdf.title = pdf_name
        self.pdf.verbose = verbose
        self.pdf.waitUntilStart()
        self.pdf.enqueueTitleSection(device=self.device.azdknum, date_beg=date_beg, date_end=date_end,
                                     bin_mode=self.device.flags.SCF_BIN_MODE_DIS == 0,
                                     data_freq=self.device.frequency, exp_time=self.device.exptime,
                                     fw_version=self.device.fw_version, comm_speed=comm_speed)

    def _wait_for_mode(self, wmode : WMode):
        match wmode:
            case WMode.Idle: wmodecode = 48
            case WMode.Autonomous: wmodecode = 49
            case WMode.FreeFrameRead: wmodecode = 53
            case WMode.FreeFrameReadAvg: wmodecode = 57
            case _: return
        cmd = self.azdkdb.createcmd(wmodecode, timeout=self.timeout_default*0.5)
        if not call_azdk_cmd(self.azs, cmd, self.timeout_default): return False
        while self.azs.is_alive():
            cmd = self.azdkdb.createcmd(16, timeout=self.timeout_default*0.5)
            if not call_azdk_cmd(self.azs, cmd, self.timeout_default): continue
            cmd = WState.from_bytes(cmd.answer)
            if cmd.wmode == wmode.value: return True
        return False

    def _add_stats(self, statsCmd : AzdkCmd):
        if call_azdk_cmd(self.azs, statsCmd, self.timeout_default):
            if isinstance(statsCmd.answer, bytes):
                _stats = self.azdkdb.interpret(statsCmd.code, statsCmd.answer)
                tNow = dt.datetime.now().timestamp()
                self.analyzer.addStats(tNow, _stats)
                return True
        return False

    def _start_measurements(self):
        self.pds.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(0), timeout=self.timeout_default*0.5), self.timeout_default)
        self._wait_for_mode(WMode.Idle)
        cmd = self.azdkdb.createcmd(76, [0, 67, 0, 0], timeout=self.timeout_default*0.5)
        call_azdk_cmd(self.azs, cmd, self.timeout_default)
        self.device.getinfo(self.azs, self.azdkdb)
        self.azs.clearNotifications()

    def openSockets(self, ip='127.0.0.1', azs_port=56001, pds_port=55555, verbose=True):
        self.azs = AzdkSocket(ip, azs_port, AzdkServerCommands, verbose, logger=self.logger)
        self.pds = AzdkSocket(ip, pds_port, PDSServerCommands, verbose, logger=self.logger)
        if not self.azs.waitUntilStart(): return False
        if not self.pds.waitUntilStart(): return False
        self._start_measurements()
        return True

    def setSockets(self, azdk_socket : AzdkSocket, pds_socket : AzdkSocket, verbose=True):
        if not isinstance(azdk_socket, AzdkSocket): return False
        if not isinstance(pds_socket, AzdkSocket): return False
        if not azdk_socket.waitStarted(): return False
        if not pds_socket.waitStarted(): return False
        self.azs = azdk_socket
        self.pds = pds_socket
        self.azs.verbose = verbose
        self.pds.verbose = verbose
        self._start_measurements()
        return True

    def _newRandomTrack(self, angvel : float , duration : float):
        initQuat = Quaternion.random()
        initAngVel = Vector.randomdir() * (angvel*np.pi/180)

        devAngVel = initQuat.rotateInverse(initAngVel).data()*(2**30)
        devAngVel = devAngVel.astype(np.int32)

        cmd = self.azdkdb.createcmd(7, devAngVel, timeout=self.timeout_default*0.5)
        call_azdk_cmd(self.azs, cmd, self.timeout_default)

        self.pds.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(0), timeout=self.timeout_default*0.5), self.timeout_default)

        pdsSetQuat = PDSServerCmd(PDSServerCommands.SET_ORIENT, initQuat, timeout=self.timeout_default*0.5)
        self.pds.execute(pdsSetQuat, self.timeout_default)

        pdsSetAngVel = PDSServerCmd(PDSServerCommands.SET_ANGVEL, initAngVel, timeout=self.timeout_default*0.5)
        self.pds.execute(pdsSetAngVel, self.timeout_default)

        fl = PDSServerCommands.STATE_ROTATION_ON.value | PDSServerCommands.STATE_SHOW_ON.value
        cmd = PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(fl), timeout=self.timeout_default*0.5)
        if not self.pds.execute(cmd, self.timeout_default*2):
            raise TimeoutError('Cannot start new track')

        k = len(self.analyzer.tracks)

        track = self.analyzer.getTrack(k - 1)
        if track is not None:
            track.duration = cmd.time_ex.timestamp()-track.tstart
        #    self.analyzer.addQuat(track.getQuat(cmd.time_ex), cmd.time_ex)

        track = self.analyzer.addTrack(initQuat, initAngVel, cmd.time_ex, f'v_{k}={angvel}')
        track.duration = duration

        #if track is not None:
        #    self.analyzer.addQuat(track.getQuat(cmd.time_ex), cmd.time_ex)

        self.log(f'New track: {track}')

        
        self.azs.clearNotifications()

        return track

    def doRandomTracks(self, angvel=0.0, count=30, duration=120.0):
        self._wait_for_mode(WMode.Autonomous)

        self.count = count

        self.analyzer._prepareTempData(int(count*duration*self.device.frequency),
                                       int((count*duration)//self.statsRequestInterval))

        track = self._newRandomTrack(angvel,duration=duration)
        tStart = time.perf_counter()
        kCur = 0

        statsTimer = tStart
        statsCmd = self.azdkdb.createcmd(26, timeout=self.timeout_default*0.5)

        pbar = None if self.azs.verbose else tqdm(desc=f'AzdkFunTest-{angvel:.1f}', total=count*duration, bar_format='{l_bar}{bar}')

        self._add_stats(statsCmd)

        while self.pds.is_alive() and self.azs.is_alive():
            time.sleep(0.01)
            t = time.perf_counter()
            if pbar:
                pbar.n = min(kCur*duration + t - tStart, pbar.total)
                pbar.update(0)
            if t > tStart + duration:
                kCur += 1
                if kCur >= count: break
                track = self._newRandomTrack(angvel,duration=duration)
                tStart = time.perf_counter()
            if t > statsTimer + self.statsRequestInterval:
                if self._add_stats(statsCmd):
                    statsTimer = t
            while len(self.azs.notifications) > 0:
                cmd = self.azs.getNotification()
                if cmd.iserror: continue
                if cmd.code == AzdkServerCommands.DEVICE_CMD.code and cmd.answer[0] == 114:
                    if isinstance(cmd.answer[1] , bytes):
                        q, t = self.azdkdb.interpret(114, cmd.answer[1])
                        self.analyzer.addQuat(q, t)

        self._add_stats(statsCmd)

        self.pds.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(0), timeout=self.timeout_default*0.5), self.timeout_default)
        self._wait_for_mode(WMode.Idle)

        self.analyzer._finishTempData()

        if pbar: pbar.close()

    def doOrbital(self, duration=3600.0):
        self._wait_for_mode(WMode.Autonomous)

        pbar = None if self.azs.verbose else tqdm(desc='AzdkFunTest-Orbital', total=duration*self.device.frequency, bar_format='{l_bar}{bar}')

        self.analyzer._prepareTempData(int(duration*self.device.frequency),
                                       int(duration//self.statsRequestInterval))

        fl = PDSServerCommands.STATE_ROTATION_ON.value | PDSServerCommands.STATE_SHOW_ON.value | PDSServerCommands.STATE_ORBITAL_ON.value
        self.pds.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(fl), timeout=self.timeout_default*0.5), self.timeout_default)
        tStart = time.perf_counter()

        statsTimer = tStart
        statsCmd = self.azdkdb.createcmd(26, timeout=self.timeout_default*0.5)

        while self.pds.is_alive() and self.azs.is_alive():
            time.sleep(0.01)
            dt = time.perf_counter() - tStart
            if dt > duration: break
            if pbar:
                pbar.n = int(dt*self.device.frequency)
                pbar.update(0)
            cmd = self.pds.getNotification()
            if cmd is not None and cmd.code == PDSServerCommands.GET_ORIENT.code:
                self.analyzer.addTrack(cmd.answer[0], t_start=cmd.time_ex)
            while len(self.azs.notifications) > 0:
                cmd = self.azs.getNotification()
                if cmd.iserror: continue
                if cmd.code == AzdkServerCommands.DEVICE_CMD.code and cmd.params[0] == 114:
                    q, t = self.azdkdb.interpret(114, cmd.params[1])
                    self.analyzer.addQuat(q, t)
            if t > statsTimer + self.statsRequestInterval:
                if call_azdk_cmd(self.azs, statsCmd, self.timeout_default):
                    _stats = self.azdkdb.interpret(statsCmd.code, statsCmd.answer)
                    tNow = dt.datetime.now().timestamp()
                    self.analyzer.addStats(tNow, _stats)
                    statsTimer = t

        self.pds.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(0), timeout=self.timeout_default*0.5), self.timeout_default)
        self._wait_for_mode(WMode.Idle)

        self.analyzer._finishTempData()

        if pbar: pbar.close()

    def analyzeResults(self, sfx = ''):
        wdir = self.wdir + f'/{self.device.azdknum}{sfx}/'

        angvel = self.analyzer.tracks[-1].rotator.speed * (180/np.pi)
        count = len(self.analyzer.tracks)
        duration = self.analyzer.tracks[-1].duration

        imgdir = wdir if self.save_figures else None
        self.analyzer.addQuatsFigure(True, imgdir, sfx != 'ss')
        self.analyzer.addDeltaAnglesFigures(imgdir=imgdir)
        self.analyzer.addErrorHist(imgdir=imgdir, addToReport=angvel < 1)
        self.analyzer.addStatsFigure(True, imgdir, cols=[1,2,3,5])
        if angvel == 0.0:
            err_file = imgdir + '/error.csv' if imgdir else None
            errs = None if sfx == 'ss' else self.analyzer.calcErrors(err_file)
            self.pdf.enqueueAccuracySection(count=count, duration=duration,
                                            images=self.analyzer.figures, datafile=errs)
        else:
            _stats = self.analyzer.stats
            kh = _stats.head(1)
            ke = _stats.tail(1)
            _stats = [int(k2) - int(k1) for k1, k2 in zip(kh._values[0][1:], ke._values[0][1:])]
            self.pdf.enqueueStatisticsSection(count=self.count, duration=duration,
                                              images=self.analyzer.figures, stats=_stats,
                                              angvel=angvel)
        self.pdf.sync()
        self.analyzer.figures.clear()

    def finish(self):
        self.pdf.enqueueFinish()
        self.pdf.sync(True)
        self.analyzer.stop()
        self.azs.stop()
        self.pds.stop()

def azdk_fun_tests(*, count=30, duration=120.0, wdir : str = None, angvels : dict = None,
                   azs : AzdkSocket = None, pds : AzdkSocket = None, verbose=False, ip='127.0.0.1',
                   azdkdbpath = 'AZDKHost.xml'):
    bot = None
    #bot = TeleBot("5811298447:AAF0--61uBVvKgFvMeYs76fB1QjmhaihU-Y", -822387173)

    if wdir is None:
        wdir = os.path.dirname(__file__)

    if angvels is None:
        angvels = {'s': 0.0, 'r01': 0.1, 'r1': 1.0, 'r2': 2.0, 'r3': 3.0}

    # setup PDF
    fte = FunTestExecutor(wdir, azdkdbpath=azdkdbpath, save_figures=False)
    if not fte.setSockets(azs, pds, verbose) and not fte.openSockets(ip=ip, verbose=verbose):
        print('Failed: cannot open sockets')
        return
    fte.restartPdfDaemon()

    for sfx, angvel in angvels.items():
        _cnt, _dur = count, duration
        if isinstance(angvel, tuple|list):
            if len(angvel) > 1: _cnt = angvel[1]
            if len(angvel) > 2: _dur = angvel[2]
            if len(angvel) > 0: angvel = angvel[0]
        else:
            angvel = float(angvel)
        if bot: bot.message(f"Запуск функционального теста '{sfx}' {fte.device.devname}")
        fte.doRandomTracks(angvel, _cnt, _dur)
        fte.analyzeResults(sfx)
        fte.logger.flush()
        if bot: bot.message(f"Завершение теста '{sfx}' {fte.device.devname}")

    fte.finish()
    fte.logger.flush()
    if bot: bot.sendfile(fte.pdf.autosave_path, f'Результаты тестирования {fte.device.devname}')

if __name__ == "__main__":
    #azdk_fun_tests(count=2, duration=5.0, angvels={'s': 0.0, 'r01': 0.1, 'r1': 1.0})
    azdk_fun_tests(duration = 5,count=5,angvels={'s': 0.0},wdir="D:/AZDK/programs/AzdkSoft/scripts/presset/wdir",
                   azdkdbpath="D:/AZDK/programs/AzdkSoft/scripts/presset/AZDKHost.xml",ip="25.21.118.38")
    os.system('pause')
