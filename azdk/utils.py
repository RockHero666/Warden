import os
import io
import re
import time
from threading import Thread, Lock
import datetime as dt
from pandas import read_csv
import numpy as np
try: from tqdm import tqdm
except ImportError: tqdm = None
import sys

class RegExp:
    reDbl = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    reInt = re.compile(r'(\d+)')
    reTime = re.compile(r'(\d{2,2})\:(\d{2,2})\:(\d{2,2})(\.\d+)?')
    reDate = re.compile(r'(\d{2,2})[-.](\d{2,2})[-.](\d{4,4})')
    reDateAlt = re.compile(r'(\d{4,4})[-.](\d{2,2})[-.](\d{2,2})')
    reQuat = re.compile(r'\{\s*' + reDbl + r',\s*' + reDbl + r',\s*' + reDbl + r',\s*' + reDbl + r'\s*\}')
    reVec = re.compile(r'\{\s*' + reDbl + r',\s*' + reDbl + r',\s*' + reDbl + r'\}')
    reDbls = re.compile(reDbl)
    reHex1 = re.compile(r'0[xX]([0-9a-fA-F]+)')
    reHex2 = re.compile(r'([\0-9a-fA-F]+)h')
    rePhc = re.compile(reDbl + r'[^0-9.]+' + reDbl + r'[^0-9.]+' + reDbl + r'\D+(\d+)')
    reSupSub = re.compile(r'([^^_]*)(([\^_])([^^_]))?')

class AzdkLogger:
    def __init__(self, filepath : str = None, timestamp_fl=True):
        self.timestamp_fl = timestamp_fl
        if isinstance(filepath, str):
            self.fl = open(filepath, 'at', encoding='utf-8')
        elif isinstance(filepath, io.TextIOWrapper):
            self.fl = filepath
        else:
            self.fl = None
            #raise ValueError('Wrong type: filepath')

    def log(self, txt : str):
        _timestamp = dt.datetime.now().strftime(r'%Y.%d.%m %H:%M:%S.%f') if self.timestamp_fl else ''
        #self.fl.write(_timestamp + txt + '\n')
        print(_timestamp + txt, file=self.fl)

    def flush(self):
        self.fl.flush()

class AzdkThread(Thread):
    def __init__(self, name : str = None, verbose=False, logger : AzdkLogger = None):
        super().__init__(name=name or self.__class__.__name__)
        self._mutex = Lock()
        self.running = False
        self.verbose = verbose
        if isinstance(logger, io.TextIOWrapper | str):
            self.logger = AzdkLogger(logger)
        else:
            self.logger = logger
        self.errmsg = None

    def stop(self, sync=True):
        self.running = False
        if sync: self.join()

    def __enter__(self):
        if self.verbose:
            self.log(f'Starting {self.name}')
        self.start()
        self.waitStarted()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.verbose: self.log(f'Stopping {self.name}')
        self.stop()

    def log(self, txt : str, toconsole=True, tofile=True):
        tofile = tofile and self.logger is not None
        if not tofile and not toconsole: return
        txt = f'{self.name}; ' + txt
        if tofile: self.logger.log(txt)
        if toconsole: print(txt)

    def waitStarted(self, timeout=1.0):
        ts = time.perf_counter()
        while not self.is_alive() or not self.running:
            time.sleep(0.01)
            if timeout > 0 and time.perf_counter() > ts + timeout:
                return False
        return True

    def waitUntilStart(self, timeout=1.0):
        self.start()
        return self.waitStarted(timeout)

def getLogFileDate(datafile : str, max_line=0x7fffffff):
    date = dt.datetime.fromtimestamp(os.path.getmtime(datafile)).date()
    size = os.path.getsize(datafile)
    line_counter = 0
    with open(datafile, 'rt', encoding='UTF-8') as fl:
        for ln in fl:
            line_counter += 1
            ln = ln.strip()
            if len(ln) < 0: continue
            if ln.startswith('---==='):
                m_date = RegExp.reDate.search(ln)
                if m_date:
                    date = dt.date(int(m_date.group(3)), int(m_date.group(2)), int(m_date.group(1)))
                    break
            if line_counter > max_line: break
    return date, size

def readLogFileData(datafile : str, colnames=None, timecol=0, datacol=-1, dataconv=None, nrows=None):
    try:
        dtable = read_csv(datafile, header=None, skiprows=2, sep=';', names=colnames, nrows=nrows)
    except FileNotFoundError as e:
        print(e)
        return None
    date, _ = getLogFileDate(datafile)
    dtable = dtable[dtable.iloc[:,timecol].notna()]
    t0 = get_datetime(dtable.iloc[0, timecol], date, convert=False)
    dtable.iloc[:,timecol] = dtable.iloc[:, timecol].apply(lambda x: get_datetime(x, date, t0))
    if datacol >= 0 and dataconv:
        dtable.iloc[:, datacol] = dtable.iloc[:, datacol].apply(dataconv)
    return dtable

def to_hm_str(duration : float | dt.timedelta):
    if isinstance(duration, dt.timedelta):
        duration = duration.total_seconds()
    else:
        duration = float(duration)

    h = int(duration // 3600)
    m = int((duration + 30) // 60 - h*60)
    if m > 0:
        mm = m % 10 if m < 10 or m > 20 else 0
        match mm:
            case 1:     m = f' {m} минуту'
            case 2|3|4: m = f' {m} минуты'
            case _:     m = f' {m} минут'
    else: m =''
    if h > 0:
        match h % 10 if h < 10 or h > 20 else 0:
            case 1: m = f'{h} час{m}'
            case 2|3|4: m = f'{h} часа{m}'
            case _: m = f'{h} часов{m}'
    return m

def getdatetime(timestamp: str,  _default = None, returnLastIdx=False, omitDateRe=False):
    if omitDateRe: m_date = None
    else:
        m_date = RegExp.reDate.search(timestamp)
        if m_date is None:
            m_date = RegExp.reDateAlt.search(timestamp)
    if m_date is None:
        if isinstance(_default, dt.date):
            d = _default
        elif isinstance(_default, dt.datetime):
            d = _default.date()
        else: d = dt.date.today()
        d = [d.year, d.month, d.day]
    else:
        d = m_date.groups()
    m_time = RegExp.reTime.search(timestamp)
    mcs = 0
    if m_time is None:
        #t = _default.time() if isinstance(_default, datetime) else _default
        #if isinstance(t, time):
        #    mcs = t.microsecond
        #    t = [t.hour, t.minute, t.second]
        #else: t = [0, 0, 0]
        if returnLastIdx: return None, -1
        else: return None
    else:
        t = m_time.groups()
        if len(t) > 3 and t[3] is not None: mcs = int(float(t[3])*1e6)
        t = [int(x) for x in t[:3]]
    d = [int(x) for x in d]
    if d[2] > 1000: d = (d[2], d[1], d[0])
    d = dt.datetime(d[0], d[1], d[2], t[0], t[1], t[2], mcs)
    if returnLastIdx: return d, m_time.end() + 1
    else: return d

def correct_angles(angles: np.ndarray):
    dd = np.diff(angles)
    indicesUp,  = np.where(dd > 270)
    indicesDn,  = np.where(dd < -270)
    for kUp, kDn in zip(indicesUp, indicesDn):
        if kUp < kDn:
            angles[kUp+1:kDn+1] -= 360
        else:
            angles[kDn+1:kUp+1] += 360
    return angles

def gettimeofday(timestamp: str):
    m_time = RegExp.reTime.search(timestamp)
    if m_time is None: raise ValueError()
    t = m_time.groups()
    sec = float(t[2])
    if len(t) > 3 and t[3] is not None: sec += float(t[3])
    return (float(t[0]) + float(t[1])/60 + sec/3600)/24

def setup_ticks(axes, vmin: float, vmax: float, dmajor: float = None, dminor: float = None, *,
                majcount=None, mincount=None, isyaxis=True, isrelative=True, issymmetric=False):
    if isinstance(vmin, dt.datetime) and isinstance(vmax, dt.datetime):
        vmax = vmax.timestamp()
        vmin = vmin.timestamp()
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        elif vmax == vmin:
            raise ValueError('Starting and ending date are the same')
        if majcount:
            while vmax - vmin < majcount * dmajor:
                dmajor /= 60
                dminor /= 60
            while vmax - vmin > majcount * dmajor:
                dmajor *= 2
                dminor *= 2
        tdelta = vmax - vmin
        vmin = dt.datetime.fromtimestamp(np.floor(vmin / dmajor) * dmajor)
        vmax = dt.datetime.fromtimestamp(np.ceil(vmax / dmajor) * dmajor)
        vmajorticks = [vmin + dt.timedelta(0, x) for x in np.arange(0, tdelta, dmajor)]
        vminorticks = [vmin + dt.timedelta(0, x) for x in np.arange(0, tdelta, dminor)]
        if dmajor % 60 == 0:
            xlabels = [x.strftime('%H:%M') for x in vmajorticks]
        else:
            xlabels = [str(x.time()) for x in vmajorticks]
    else:
        vmax = float(vmax)
        vmin = float(vmin)
        if isrelative:
            delta = np.floor(np.log10(np.fabs(vmax - vmin)*0.1) + 0.5)
            delta = 10**delta
            if vmax < vmin:
                vmax = np.floor(vmax / delta) * delta
                vmin = np.ceil(vmin / delta) * delta
            else:
                vmax = np.ceil(vmax / delta) * delta
                vmin = np.floor(vmin / delta) * delta
            if issymmetric and np.sign(vmin) != np.sign(vmax):
                vv = max(np.fabs([vmin, vmax]))
                vmin = np.sign(vmin)*vv
                vmax = np.sign(vmax)*vv
            if dmajor is None: dmajor = delta
            if dminor is None: dminor = dmajor*0.25
            if majcount:
                dv = (vmax - vmin) / (majcount - 1)
            else:
                majcount = int(np.fabs(vmax - vmin)/dmajor + 0.5) + 1
                dv =  (vmax - vmin) / (majcount - 1)
                while majcount > 12:
                    majcount = ((majcount - 1) // 2) + 1
                    dv *= 2
            vmajorticks = [*np.arange(vmin, vmax + dv*0.5, dv)]
            #vmajorticks = [x*dv + vmin for x in range(majcount)]
            if mincount:
                mincount = (majcount - 1)*mincount + 1
                dv = (vmax - vmin) / (mincount - 1)
            else:
                mincount = int(np.fabs(vmax - vmin)/dminor + 0.5) + 1
                dv =  (vmax - vmin) / (mincount - 1)
            vminorticks = [*np.arange(vmin, vmax + dv*0.5, dv)]
            #vminorticks = [x*dv + vmin for x in range(mincount)]
        else:
            sx = int(np.sign(vmax - vmin))
            vmajorticks = [x*dmajor for x in range(int(vmin/dmajor),
                           int(np.ceil(vmax/dmajor)) + sx, sx)]
            vminorticks = [x*dminor for x in range(int(vmin/dminor),
                           int(np.ceil(vmax/dminor)) + sx, sx)]
        xlabels = None
    if not isinstance(axes, list) and not isinstance(axes, np.ndarray):
        axes = [axes]
    if isyaxis:
        for ax in axes:
            #ax.ticklabel_format(useOffset=False)
            ax.set_ylim(vmin, vmax)
            ax.set_yticks(vmajorticks)
            ax.set_yticks(vminorticks, minor=True)
    else:
        for ax in axes:
            ax.set_xlim(vmin, vmax)
            ax.set_xticks(vmajorticks)
            ax.set_xticks(vminorticks, minor=True)
            if xlabels is not None:
                ax.set_xticklabels(xlabels)

def setup_yticks(axes, data, majcountlimit=None):
    ymin = np.nanmin(data, 0)
    ymax = np.nanmax(data, 0)
    n = min(len(axes), len(ymin), len(ymax))

    for k in range(n):
        dmax = np.abs(ymax[k] - ymin[k])
        dexp = np.ceil(np.log10(dmax)) - 1
        setup_ticks(axes[k], ymin[k], ymax[k], isrelative=False, dmajor=10**dexp,
                    dminor=10**(dexp-1), majcount=majcountlimit)

def _buildMaxHeapIndices(arr, _pbar : tqdm = None):
    """ Build max heap from an array
    """
    n = len(arr)
    indices = [x for x in range(n)]
    for k in range(n):
        if _pbar: _pbar.update()
        ii = indices[k]
        hi = indices[int((k - 1) / 2)]
        if arr[ii] > arr[hi]:
            j = k
            jh = int((j - 1) / 2)
            while arr[indices[j]] > arr[indices[jh]]:
                indices[j], indices[jh] = indices[jh], indices[j]
                j = jh
                jh = int((j - 1) / 2)
    return indices

def heapsort(arr, *, verbose=False):
    """ Heap sort function. Doesn't change source array.
        Return indices of source array to build sorted array.
    """
    pbar = None
    n = len(arr)
    if verbose: print(f'Heap sorting on {n} elements')
    if verbose and tqdm: pbar = tqdm(total=2*n)

    indices = _buildMaxHeapIndices(arr, pbar)

    for i in range(n - 1, 0, -1):
        indices[0], indices[i] = indices[i], indices[0]
        j, index = 0, 0
        while True:
            index = 2 * j + 1
            if index < i - 1 and arr[indices[index]] < arr[indices[index + 1]]:
                index += 1
            if index < i and arr[indices[j]] < arr[indices[index]]:
                indices[j], indices[index] = indices[index], indices[j]
            j = index
            if index >= i: break
        if pbar: pbar.update()
    if pbar: pbar.close()
    return indices

def get_datetime(t_str : str, date : dt.date = None, check_dt : dt.datetime = None, convert=True):
    m_time = RegExp.reTime.search(t_str)
    t = None
    if m_time:
        g = m_time.groups()
        t = dt.time(int(g[0]), int(g[1]), int(g[2]), int(1e6*float(g[3])))
        if isinstance(date, dt.date):
            t = dt.datetime.combine(date, t)
            if isinstance(check_dt, dt.datetime) and t < check_dt:
                t += dt.timedelta(days=1)
            if convert: t = t.timestamp()
    return t

if __name__ == "__main__":
    pass
