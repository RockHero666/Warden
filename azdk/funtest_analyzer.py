import os
import datetime as dt
from enum import Enum
from bisect import bisect_right
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from matplotlib import pyplot as pp
from azdk.utils import RegExp, AzdkThread, setup_ticks, readLogFileData
from azdk.linalg import Quaternion, Vector, Rotator
from azdk.pdfcreator import FunTestPdfCreator, PdfFunTestReport

def get_max_durations(times : list):
    dtimes = [np.round((t2-t1).total_seconds()) for t1, t2 in zip(times, times[1:])]
    dtimes = [x for x in dtimes if x > 0]
    dtimes = np.array(dtimes)
    dtimes_set = np.unique(dtimes, return_counts=True)
    duration = np.max(dtimes_set)
    indices = np.where(dtimes == duration)
    return duration, indices[0]

def get_quat(q_str : str):
    m_quat = RegExp.reQuat.search(q_str)
    q = None
    if m_quat:
        g = m_quat.groups()
        q = Quaternion(float(g[0]), float(g[1]), float(g[2]), float(g[3]))
    return q

def get_vec(v_str : str):
    m_vec = RegExp.reVec.search(v_str)
    v = None
    if m_vec:
        g = m_vec.groups()
        v = Vector(float(g[0]), float(g[1]), float(g[2]))
    return v

def get_dbls(_str : str):
    dbls = RegExp.reDbls.findall(_str)
    return np.array([float(x) for x in dbls])

def get_quat_or_vec(_str : str):
    val = get_quat(_str)
    if val is None: val = get_vec(_str)
    return val

def quat_diff_angles(qo : Quaternion, qc : Quaternion):
    dq = qo.conjugated() * qc
    vxy = dq.rotate(Vector(0,0,1))
    vz = dq.rotate(Vector(1,0,0))
    xa = np.arctan(vxy.x) * 206264.8
    ya = np.arctan(vxy.y) * 206264.8
    za = np.arctan2(vz.y, vz.x) * 206264.8
    return np.array((xa, ya, za))

def check_dir(dirpath):
    try: os.mkdir(dirpath)
    except FileExistsError: pass
    if not dirpath.endswith('/'): dirpath += '/'
    return dirpath

class FigureTypes(Enum):
    COMPONENTS = 'wxyz'
    QUATS = 'Компонент кватерниона {c}, измеренного в МЗД (красные точки) и компонент кватерниона, установленного в ОДС (синяя линия)'
    ANG_DIFFS = 'Разница углов ориентации между измерениями в МЗД и установкой в ОДС в ПСК'
    ANG_HIST = 'Гистограммы ошибок углов ориентации Δx, Δy, Δz, представленные в системе координат прибора'
    ANG_DIFFS_AVG = 'Разница углов ориентации между измерениями в МЗД вычисленными средними значениями в ПСК'
    ANG_HIST_AVG = 'Гистограммы отклонений от средних значений углов ориентации Δx, Δy, Δz, представленные в ПСК'
    FREQS = 'Частота выдачи МЗД АЗДК-1 кадров по статистическим данным. Красными точками обозначены «плохие» кадры, синими – прочитанные кадры, зелеными – кадры с успешными отождествлениями'

class PdsTrack():
    def __init__(self, angvel : Vector, initquat : Quaternion, tstart : float = None, duration : float = None):
        if tstart is None:
            tstart = dt.datetime.now().timestamp()
        elif isinstance(tstart, dt.datetime):
            tstart = tstart.timestamp()
        else:
            tstart = float(tstart)
        self.rotator = Rotator(angvel, initquat)
        self.tstart = tstart
        self.name = None
        self.quats = []
        self.times = []
        self.diffs = []
        self.duration = duration
        self.idx = 0

    def addQuat(self, quat_o : Quaternion, time_o : dt.datetime | float):
        if isinstance(time_o, dt.datetime):
            time_o = time_o.timestamp()
        if self.cmpTime(time_o): return None
        quat_c = self.rotator.rotation(time_o - self.tstart)
        diffs = quat_diff_angles(quat_o, quat_c)
        absval = np.sqrt(sum(diffs**2))
        if absval > 3600: return None
        if quat_o.w < 0: quat_o.inverse()
        self.quats.append(quat_o)
        self.times.append(time_o)
        self.diffs.append(diffs)
        return time_o, quat_o, quat_c, diffs

    def cmpTime(self, t : float):
        dt = t - self.tstart
        if dt < 0.0: return -1
        if dt > self.duration: return + 1
        return 0

    def __getitem__(self, idx : int):
        return self.times[idx], self.quats[idx]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        k = self.idx
        self.idx = k + 1
        if self.idx >= len(self.quats):
            raise StopIteration
        qo = self.quats[k]
        t = self.times[k]
        qc = self.rotator.rotation(t - self.tstart)
        da = quat_diff_angles(qo, qc)
        return t, qo, da

    def __str__(self) -> str:
        s = f'qs={self.rotator.qs}, vd={self.rotator.dir}, vs={self.rotator.speed*180/np.pi:.4f}'
        s += f', t={dt.datetime.fromtimestamp(self.tstart)}'
        if self.duration:
            s += f', dt={self.duration:.2f}'
        return s

    def getQuat(self, q_time : float):
        if isinstance(q_time, dt.datetime):
            q_time = q_time.timestamp()
        return self.rotator.rotation(q_time - self.tstart)

class FunTestAnalyzer(AzdkThread):
    class StatCols(Enum):
        time = 0
        total = 1
        read = 2
        bad = 3
        processsed = 4
        identified = 5
        ii = 6
        am = 7
        skipped = 8
        catalog = 9

    class DataCols(Enum):
        time = 0
        w = 1
        x = 2
        y = 3
        z = 4
        xerr = 5
        yerr = 6
        zerr = 7
        track = 8

    class Colors(Enum):
        gray = 0x808080
        blue = 0x0000FF
        red = 0xFF0000
        green = 0x008000
        yellow = 0x00FFFF
        cyan = 0xFFFF00
        magenta = 0xFF00FF
        gold = 0x00D7FF
        purple = 0x800080
        lime = 0x00FF00
        navy = 0x000080

    class ErrorCols(Enum):
        count = 0
        qw = 1
        qx = 2
        qy = 3
        qz = 4
        ra = 5
        de = 6
        phi = 7
        xa_sh = 8
        ya_sh = 9
        za_sh = 10
        xa_std = 11
        ya_std = 12
        za_std = 13

    stats_columns = [x.name for x in StatCols]
    data_columns = [x.name for x in DataCols]

    def __init__(self, verbose=False, logger=None):
        super().__init__(self.__class__.__name__, verbose, logger)
        self.tracks = []
        self.stats = DataFrame(columns=self.stats_columns, dtype=np.float64)
        self.freq = []
        self.figures = []
        self.data = DataFrame(columns=self.data_columns, dtype=np.float64)
        self.data_t = np.ndarray([0, len(self.data_columns)])
        self.pdata_t = np.ndarray([0, 4])
        self.row_t = 0
        self.row_stats = 0
        self.fig_dpi = 150
        self.fig_size = [12, 7]
        self.fig_size_sm = [12, 5]
        self.fig_size_sq = [12, 12]
        pp.rcParams.update({'figure.max_open_warning': 0})

    def _findTrack(self, _time : float):
        if _time < self.tracks[0].tstart: return -1
        k = bisect_right(self.tracks, _time, key=lambda t: t.tstart) - 1
        tEnd = self[k].tstart + self[k].duration
        if tEnd is not None and _time > tEnd:
            return -1
        return k

    def getTrack(self, k) -> PdsTrack | None:
        return self.tracks[k] if k >= 0 and k < len(self.tracks) else None

    def __getitem__(self, k) -> PdsTrack:
        return self.tracks[k]

    @classmethod
    def _getbins(cls, s: float, m=0.0):
        _bins = list([-5*s, -4*s, -3*s, -2.5*s, -2*s])
        _bins += list(np.arange(-s*1.8, s*1.81, s*0.2))
        _bins += list([2*s, 2.5*s, 3*s, 4*s, 5*s])
        _bins = [x + m for x in _bins]
        return _bins

    def loadTracks(self, datafile : str, angvel=0.0, min_duration=10.0):
        dtable = readLogFileData(datafile, ['time', 'data'])
        if dtable is None: return False
        data = dtable.data.apply(get_quat_or_vec)
        tt = dtable.time.diff().apply(np.round)
        q, t, v = None, None, None if angvel else Vector()
        self.tracks.clear()
        for k, td in enumerate(tt):
            tNew = dtable.time[k]
            if td > min_duration and v is not None and q is not None and t is not None:
                self.log(f'Add track: {v}, {q}, {dt.datetime.fromtimestamp(t)}, {td}', toconsole=False)
                track = PdsTrack(v, q, t, tNew - 0.1)
                track.duration = td
                self.tracks.append(track)
            val = data[k]
            if isinstance(val, Quaternion): q = val
            elif isinstance(val, Vector): v = val
            t = tNew
        self.log(f'{len(self.tracks)} tracks has been loaded')
        return True

    def loadQuats(self, datafile : str, drop_bad_data=True, remove_empty_tracks=True):
        if len(self.tracks) == 0:
            self.log('First, load track data', tofile=False)
            return False

        self.log('Loading quaternion data')
        qtable = readLogFileData(datafile, ['time', 'cmd', 'quat', 'qtime'], 3, 2, get_quat)
        if qtable is None: return False
        qtable = qtable.reset_index()

        data = DataFrame(np.zeros((len(qtable), len(self.data_columns))), columns=self.data_columns)
        data.time = qtable.qtime

        if remove_empty_tracks:
            t0 = data.time[0] - 1.0
            k = 0
            for t in self.tracks:
                if t.tstart > t0: break
                k += 1
            self.tracks = self.tracks[k:]

        n = len(data)
        single_pass = n < 30000 and len(self.tracks) < 100
        cols = [self.DataCols.track.name, self.DataCols.time.name]
        #q_data = DataFrame(np.zeros((n, 2)), columns=['qo', 'qc'])
        q_data = DataFrame(qtable.quat, columns=['qc'])
        q_data['qo'] = qtable.quat

        if single_pass:
            self.log('  binding tracks', tofile=False)
            data.track = data.time.apply(self._findTrack)
            q_data.qc = data[cols].apply( lambda x : self.tracks[int(x.track)].getQuat(x.time), axis=1)
        else:
            pbar = tqdm(desc='Binding tracks', bar_format='{l_bar}{bar}', total=n)
            for k in range(0, n, 1000):
                kEnd = min(k + 1000, n)
                data.track.iloc[k:kEnd] = data.time.iloc[k:kEnd].apply(self._findTrack)
                q_data.qc[k:kEnd] = data.loc[k:kEnd, cols].apply(
                    lambda x : self.tracks[int(x.track)].getQuat(x.time), axis=1
                )
                pbar.update(1000)
            pbar.close()

        self.log('  extracting columns', tofile=False)
        wxyz = self.data_columns[1:5]
        data.loc[:, wxyz] = DataFrame(qtable.quat.apply(lambda x: x.data()).to_list(), columns=wxyz)

        self.log('  calculating deviations', tofile=False)
        xyz = self.data_columns[5:8]
        if single_pass:
            ang_diffs = q_data.apply(lambda x: quat_diff_angles(x.qo, x.qc), axis=1)
            data.loc[:, xyz] = DataFrame(ang_diffs.to_list(), columns=xyz)
        else:
            pbar = tqdm(desc='Calc deviations', bar_format='{l_bar}{bar}', total=n)
            for k in range(0, n, 1000):
                kEnd = min(k + 1000, n)
                ang_diffs = q_data.iloc[k:kEnd].apply(lambda x: quat_diff_angles(x.qo, x.qc), axis=1)
                data.loc[k:kEnd, xyz] = DataFrame(ang_diffs.to_list(), columns=xyz)
                pbar.update(1000)
            pbar.close()

        if drop_bad_data:
            _errs = np.abs(data.xerr) > 3600
            if len(_errs) > 0:
                _errs = data[_errs].index
                self.log(f'  droppping {len(_errs)} data', tofile=False)
                data = data.drop(_errs)

        self.log('  completed', tofile=False)
        self.log(f'{len(data)} out of {len(qtable)} quaternions has been added')

        self.data = data
        return True

    def loadStats(self, datafile : str):
        dtable = readLogFileData(datafile, self.stats_columns)
        if dtable is None: return False
        self.log(f'{len(dtable)} statistics has been loaded')
        self.stats = dtable
        return True

    def loadDataFiles(self, wdir : str, device : int, sfx : str, pattern = '.tracked.azdk'):
        ok1 = self.loadStats(wdir + f'azdkclient{pattern}{device}{sfx}.txt')
        ok2 = self.loadTracks(wdir + f'pdsserver{pattern}{device}{sfx}.txt')
        ok3 = self.loadQuats(wdir + f'azdkserver{pattern}{device}{sfx}.txt')
        return ok1 and ok2 and ok3

    def reset(self):
        self.tracks.clear()
        self.figures.clear()
        self.stats = DataFrame(columns=self.stats_columns, dtype=np.float64)
        self.data = DataFrame(columns=self.data_columns, dtype=np.float64)

    def _convert_stats(self, cols, rolling_wnd):
        dtable = self.stats
        if len(self.data) > 0:
            tb = self.data.time.min()
            te = self.data.time.max()
            times = (dtable.time >= tb)*(dtable.time <= te)*(dtable.ii.notna())
            dtable = dtable.loc[times,:]
        data = []
        tdiff = dtable.time.diff()
        #tdiff = tdiff.rolling(rolling_wnd, 1, True).median()
        times = dtable.time.rolling(rolling_wnd, 1, True).mean()
        vmax = 0
        for k, col in enumerate(cols):
            if isinstance(col, int):
                col = self.stats_columns[col]
                cols[k] = col
            elif isinstance(col, self.StatCols):
                col = col.name
                cols[k] = col
            elif not isinstance(col, str):
                raise ValueError()
            ddiff = dtable[col].astype(int).diff()
            #ddiff = ddiff.rolling(rolling_wnd, 1, True).median()
            d = ddiff / tdiff
            d = d.rolling(rolling_wnd, 1, True).median()
            d += 0.01*k
            vmax = max(d.max(), vmax)
            data.append(d[1:])
        t = [dt.datetime.fromtimestamp(x) for x in times[1:]]
        return t, data, vmax

    def addTrack(self, q : Quaternion, v=Vector(), t_start=0.0, name : str = None):
        track = PdsTrack(v, q, t_start)
        track.name = name
        self.tracks.append(track)
        name = name or f'#{len(self.tracks)}'
        self.log('Adding track ' + name)
        return track

    def _prepareTempData(self, count : int, statscount : int):
        self.data_t = np.zeros([count, len(self.data_columns)])
        self.pdata_t = np.zeros([count, 4])
        self.row_t = 0
        self.tracks.clear()
        self.stats = np.zeros([statscount, len(self.stats_columns)])
        self.row_stats = 0

    def addStats(self, t : float | dt.datetime, _stats : list | np.ndarray):
        k = self.row_stats
        self.row_stats = k + 1
        if k >= len(self.stats): return
        if isinstance(t, dt.datetime):
            t = t.timestamp()
        self.stats[k, 0] = t
        self.stats[k, 1:] = _stats

    def setStats(self, _stats : list | np.ndarray):
        self.stats = DataFrame(_stats, columns=self.stats_columns)

    def _finishTempData(self):
        self.data = DataFrame(self.data_t[:self.row_t], columns=self.data_columns)
        self.stats = DataFrame(self.stats[:self.row_stats], columns=self.stats_columns)

    def _pdata(self):
        pdata = DataFrame(columns=self.data_columns[0:5])
        for track in self:
            n = int(track.rotator.speed * track.duration / np.pi * 180 + 1)
            dt = track.duration / n
            for dt in np.arange(0, track.duration + dt*0.5, dt):
                t = track.tstart + dt
                q = track.getQuat(t)
                pdata.loc[len(pdata)] = [t, q.w, q.x, q.y, q.z]
        return pdata

    def addQuat(self, q : Quaternion, t : float | dt.datetime):
        k = self.row_t
        if k >= len(self.data_t): return
        if isinstance(t, dt.datetime):
            t = t.timestamp()
        elif not isinstance(t, float|int):
            return
        t_num = len(self.tracks) - 1
        track = self.tracks[t_num]
        track.addQuat(q, t)
        qc = track.rotator.rotation(t - track.tstart)
        da = quat_diff_angles(q, qc)
        data_row = t, q.w, q.x, q.y, q.z, da[0], da[1], da[2], t_num
        #row_num = len(self.data.loc)
        #self.data.loc[row_num] = data_row
        self.data_t[k, :] = data_row
        self.pdata_t[k, :] = qc.w, qc.x, qc.y, qc.z
        self.row_t = k + 1

    def analyse(self):
        if len(self.tracks) == 0: return

        n = sum([len(x.quats) for x in self.tracks])
        pbar = tqdm(desc='Creating quats', bar_format='{l_bar}{bar}', total=n)

        data = DataFrame(columns=self.data_columns, dtype=np.float64)

        for track in self.tracks:
            for t, q, da in iter(track):
                data.loc[len(data)] = t, q.w, q.x, q.y, q.z, da[0], da[1], da[2]

        pbar.close()

        self.data = data

    def addQuatsFigure(self, legend=False, imgdir : str = None, addToReport=True):
        if len(self.data) == 0: return
        if not addToReport and imgdir is None: return

        times = [dt.datetime.fromtimestamp(x) for x in self.data.time]
        

        pdata = self._pdata()
        ptimes = pdata.time.apply(dt.datetime.fromtimestamp)

        for c in FigureTypes.COMPONENTS.value:
            name = f'quaternion.{c}'
            self.log(f"  create figure '{name}'")
            fig, ax = pp.subplots(figsize=self.fig_size, dpi=self.fig_dpi)
            ax.clear()
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(name)

            setup_ticks(ax, times[0], times[-1], 300, 60, isyaxis=False, majcount=10)
            setup_ticks(ax, 0 if c=='w' else -1, 1, 0.1, 0.05, isyaxis=True)
            ax.scatter(times, self.data[c], c='red', marker='.', label='Stratracker')

            ax.plot(ptimes, pdata[c], label='PDS values')

            if legend: fig.legend(loc='upper right', bbox_to_anchor=(1, 0.975))

            if isinstance(imgdir, str):
                imgdir = check_dir(imgdir)
                self.log(f'  saving {name}')
                fig.savefig(imgdir + name + '.png', dpi=300, format='PNG', bbox_inches='tight')

            if addToReport:
                self.figures.append((fig, FigureTypes.QUATS.value.format(c=c)))

    def addStatsFigure(self, legend=False, imgdir : str = None, cols=None, rolling_wnd=5, addToReport=True):
        if len(self.stats) == 0: return
        if not addToReport and imgdir is None: return

        figname = 'Stats frequency'
        self.log(f"  create figure '{figname}'")

        if cols is None:
            cols = self.stats_columns[1:]

        times, data, vmax = self._convert_stats(cols, rolling_wnd)

        fig, ax = pp.subplots(figsize=self.fig_size, dpi=self.fig_dpi)
        #fig.set_tight_layout({"pad": .0})
        #fig.set_visible(False)
        ax.clear()
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title(figname)
        setup_ticks(ax, times[0], times[-1], 300, 60, isyaxis=False, majcount=10)
        setup_ticks(ax, 0, np.ceil(vmax), 1, 0.1, isyaxis=True)

        for d, _clr, col in zip(data, FunTestAnalyzer.Colors, cols):
            ax.scatter(times, d, c=f'#{_clr.value:06X}', marker='.', label=col)
        if legend: fig.legend(loc='upper right', bbox_to_anchor=(1, 0.975))

        if isinstance(imgdir, str):
            imgdir = check_dir(imgdir)
            self.log(f'  saving {figname}')
            pngfilepath = imgdir + 'stats.frequency.png'
            fig.savefig(pngfilepath, dpi=300, format='PNG', bbox_inches='tight')

        if addToReport:
            self.figures.append((fig, FigureTypes.FREQS))

    def addDeltaAnglesFigures(self, ybounds=5.0, meanCentered=False, imgdir : str = None, addToReport=True):
        if len(self.data) == 0: return
        if not addToReport and imgdir is None: return

        figname = 'Angle errors'
        self.log(f"  create figure '{figname}'")

        components = FigureTypes.COMPONENTS.value[1:]
        times = [dt.datetime.fromtimestamp(x) for x in self.data.time]
        cols = [self.DataCols.xerr.name, self.DataCols.yerr.name, self.DataCols.zerr.name]

        fig, axes = pp.subplots(len(cols), 1, figsize=self.fig_size_sm, dpi=self.fig_dpi)
        setup_ticks(axes, times[0], times[-1], 120, 30, isyaxis=False, majcount=24)

        data = self.data
        #TODO mean centered errors
        if meanCentered: pass

        for ax, col, c in zip(axes, cols, components):
            if ybounds is None:
                ymin = data[col].min()
                ymax = data[col].max()
            else:
                sigma = data[col].std()
                ymax = sigma*ybounds
                if ymax > 10:
                    ymax = np.floor(ymax / 10) * 10
                ymin = -ymax
            setup_ticks(ax, ymin, ymax, 20, 5, isyaxis=True, majcount=5, mincount = 5, issymmetric=True)
            ax.set_xlabel('Time')
            ax.set_ylabel('\u0394' + c + ', \u02BA')
            ax.tick_params(labelsize=8)
            ax.scatter(times, data[col], c='red', marker='.', s=1.0)
            ax.set_title(f'Angle {c} deviations')

        if isinstance(imgdir, str):
            if meanCentered: imgdir += '/error-angles-avg.png'
            else: imgdir += '/error-angles.png'
            self.log(f'  saving {figname}')
            fig.savefig(imgdir, dpi=300, format='png', bbox_inches='tight')

        if addToReport:
            _type = FigureTypes.ANG_DIFFS_AVG if meanCentered else FigureTypes.ANG_DIFFS
            self.figures.append((fig, _type))

    def addErrorHist(self, xybounds=5.0, meanCentered=False, imgdir : str = None, addToReport=True):
        if len(self.data) == 0: return
        if not addToReport and imgdir is None: return

        figname = 'Error histograms'
        self.log(f"  create figure '{figname}'")

        data = self.data
        #TODO mean centered histograms
        if meanCentered: pass

        cols = [self.DataCols.xerr.name, self.DataCols.yerr.name, self.DataCols.zerr.name]
        stds = data[cols].std()
        means = data[cols].mean()

        #, gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
        fig, axes = pp.subplots(2, 2, figsize=self.fig_size_sq, dpi=self.fig_dpi)
        axes = [axes[0,0], axes[1,1], axes[0,1], axes[1,0]]
        components = FigureTypes.COMPONENTS.value[1:]
        ax = axes[3]
        ax.scatter(data[cols[0]], data[cols[1]], marker='.')

        xb = ax.get_xbound()
        yb = ax.get_ybound()
        xb = max(np.fabs(xb))
        yb = max(np.fabs(yb))
        xb = min(xb, stds[cols[0]]*xybounds)
        yb = min(yb, stds[cols[1]]*xybounds)
        if xb > 10: xb = np.floor(xb / 10) * 10
        if yb > 10: yb = np.floor(yb / 10) * 10
        ax.set_xbound(-xb, xb)
        ax.set_ybound(-yb, yb)
        xb = (-xb, xb, xb*0.04)
        yb = (-yb, yb, yb*0.04)

        ax.set_xlabel('\u0394' + components[0] + ', \u02BA')
        ax.set_ylabel('\u0394' + components[1] + ', \u02BA')
        ax.tick_params(labelsize=8)

        for ax, col, c in zip(axes, cols, components):
            ax.tick_params(labelsize=8)
            ori = 'horizontal' if ax == axes[1] else 'vertical'
            _bins = self._getbins(stds[col])
            ax.hist(data[col], _bins, density=1, orientation=ori)
            title = 'Histogram of \u0394' + c + r', $\sigma_' + c + f'$={stds[col]:.1f}' + '\u02BA'
            if not meanCentered:
                title += f' M={means[col]:5.1f}' + '\u02BA'
            if ax == axes[1]:
                ax.set_xlabel('Probability density')
                ax.set_ylabel(title)
                x = np.arange(yb[0] + yb[2]*0.5, yb[1], yb[2])
            else:
                ax.set_title(title)
                ax.set_ylabel('Probability density')
                if ax == axes[2]:
                    xb = np.fabs(ax.get_xbound())
                    xb = min([*xb, stds[col]*5])
                    if xb > 10: xb = np.floor(xb / 10) * 10
                    xb = (-xb, xb, xb*0.04)
                x = np.arange(xb[0] + xb[2]*0.5, xb[1], xb[2])
            if ax != axes[0]:
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
            #if hlog: ax.set_xscale('log')
            y = ((1 / (np.sqrt(2 * np.pi) * stds[col])) * np.exp(-0.5 * (1 / stds[col] * (x - means[col]))**2))
            if ax == axes[1]:
                ax.plot(y, x, '--')
                ax.set_ybound(yb[0], yb[1])
            else:
                ax.plot(x, y, '--')
                ax.set_xbound(xb[0], xb[1])

        if isinstance(imgdir, str):
            if meanCentered: imgdir += '/error-hist-avg.png'
            else: imgdir += '/error-hist.png'
            self.log(f'  saving {figname}')
            fig.savefig(imgdir, dpi=300, format='PNG', bbox_inches='tight')

        if addToReport:
            _type = FigureTypes.ANG_HIST_AVG if meanCentered else FigureTypes.ANG_HIST
            self.figures.append((fig, _type))

    def calcErrors(self, filepath: str = None):
        if len(self.tracks) < 1: return

        try: _fl = open(filepath, 'wt', encoding='UTF-8')
        except FileNotFoundError: _fl = None
        except TypeError: _fl = None

        cols = [x.name for x in self.ErrorCols]
        df = self.data.groupby(self.DataCols.track.name)
        mn_vals = df.mean()
        std_vals = df.std()
        cnt_vals = df.count()
        data = np.full((len(self.tracks), len(cols)), np.nan)
        data = DataFrame(data, columns=cols)
        for k, track in enumerate(self.tracks):
            try:
                w,x,y,z = mn_vals.loc[k, ['w','x','y','z']]
                va = Quaternion(w,x,y,z).toangles()*(180/np.pi)
                if va.x < 0: va.x += 360
                qdata = [w, x, y, z, va.x, va.y, va.z]
                x, y, z = mn_vals.loc[k, ['xerr', 'yerr', 'zerr']]
                errs = [x, y, z]
                x, y, z = std_vals.loc[k, ['xerr', 'yerr', 'zerr']]
                errs += [x, y, z]
                cnt = cnt_vals.loc[k].time
                data.iloc[k, :] = [cnt] + qdata + errs
            except KeyError:
                cnt = 0
                qdata = track.getQuat(track.tstart + track.duration*0.5)
                va = qdata.toangles()*(180/np.pi)
                if va.x < 0: va.x += 360
                errs = [np.nan]*6
                data.iloc[k, 0] = 0
                data.iloc[k, 1:5] = qdata.data()
                data.iloc[k, 5:8] = va.data()
                qdata = list(qdata.data()) + list(va.data())

            if _fl:
                s = f'{cnt};'
                s += ';'.join([f'{x:.6f}' for x in qdata])
                s += ';' + ';'.join([f'{x:.1f}' for x in errs])
                print(s, file=_fl)

        if _fl: _fl.close()

        return data

def azdk_fun_test_results_analyzer(
    device = 2317, wdir = 'd:/Users/Simae/Work/2023/sputnix',
    wdir_sfx = '', freq = 5, version = '1.05.00A5', pds_num=7, angvels=None):

    #angvels = {'s': 0.0, 'r01': 0.1, 'r1': 1.0, 'r2': 2.0, 'r3': 3.0, 'o': 0.067, 'ss': 0.0}
    if angvels is None:
        angvels = {'s': 0.0, 'r01': 0.1, 'r1': 1.0, 'r2': 2.0, 'r3': 3.0}

    pdf_name = f'ФИ АЗДК-1.5 №{device//100}-{device%100}'
    pdfd = FunTestPdfCreator(300, True, None, 'Маким Тучин', pdf_name)
    pdfd.downscale_images = False
    pdfd.autosave_path = wdir + '/' + pdf_name + wdir_sfx + '.pdf'
    if not pdfd.waitUntilStart(): return

    wdir = wdir + f'/{device}{wdir_sfx}/'

    dates = PdfFunTestReport._get_dates(wdir, device)
    pdfd.enqueueTitleSection(device=device, date_beg=dates[0], date_end=dates[1], bin_mode=freq>4,
                             data_freq=freq, exp_time=100, fw_version=version, comm_speed=500, pds_num=pds_num,
                             star_intensity = 3.0, mag_slope = 0.7, focus = 33.70)

    for sfx, angvel in angvels.items():
        #imgdir = wdir + f'azdk{device}' + sfx
        imgdir = None
        fta = FunTestAnalyzer(verbose=True)
        if not fta.loadDataFiles(wdir, device, sfx): continue
        fta.addQuatsFigure(True, imgdir, sfx != 'ss')
        fta.addDeltaAnglesFigures(imgdir=imgdir)
        fta.addErrorHist(imgdir=imgdir, addToReport=angvel < 1)
        fta.addStatsFigure(True, imgdir, cols=[1,2,3,5])

        try: duration, count = PdfFunTestReport.get_duration(wdir + f'pdsserver.tracked.azdk{device}{sfx}.txt')
        except FileNotFoundError: continue
        if angvel == 0.0:
            err_file = imgdir + '/error.csv' if imgdir else None
            errs = None if sfx == 'ss' else fta.calcErrors(err_file)
            pdfd.enqueueAccuracySection(count=count, duration=duration, images=fta.figures, datafile=errs)
        else:
            _times = PdfFunTestReport.get_time_range(wdir + f'azdkserver.tracked.azdk{device}{sfx}.txt')
            _stats = PdfFunTestReport.get_statistics(wdir + f'azdkclient.tracked.azdk{device}{sfx}.txt', *_times)
            pdfd.enqueueStatisticsSection(count=count, duration=duration, images=fta.figures, stats=_stats, angvel=angvel)
        pdfd.sync()
        fta.figures.clear()

    pdfd.enqueueFinish()
    pdfd.sync(True)
    print('finished')

if __name__ == "__main__":
    azdk_fun_test_results_analyzer(device=2327, wdir='d:/Users/Simae/Work/2023/stc', wdir_sfx='', freq=5,
                                   version='1.05.00A6', pds_num=2304)
    os.system('pause')
