"""
PGM
=====

PGM class library.
"""
from glob import glob
try: import cv2
except ImportError: cv2 = None
import numpy as np
try: from tqdm import tqdm
except ImportError: tqdm = None
#from utils import heapsort, getdatetime

class FLDMap(object):
    """ Class of old field map structure
    """
    def __init__(self, r=None):
        if isinstance(r, FLDMap):
            self._d = r._d
        elif isinstance(r, str):
            self._d = np.zeros(0)
            self.load(r)
        elif isinstance(r, np.ndarray) and len(r.shape) == 2:
            self._d = r
        else:
            self._d = np.zeros(0)

    def load(self, filepath: str):
        with open(filepath, 'rb') as fl:
            fa = np.fromfile(fl, np.uint32, 3)
            if len(fa) != 3: return False
            _cnt = fa[0]*fa[1]

            if fa[2] == 0: _type = np.uint8
            elif fa[2] == 1: _type = np.uint16
            elif fa[2] == 2: _type = np.uint32
            elif fa[2] == 3: _type = np.float64
            else: return False

            _d = np.fromfile(fl, _type, _cnt)
            if len(_d) != _cnt: return False

            #_d = _d.astype(np.float64)
            self._d = np.reshape(_d, (fa[1], fa[0]))
            return True
        return False

    def save(self, filepath: str):
        with open(filepath, 'wb') as fl:
            hdr = np.ndarray(3, np.uint32)
            hdr[0] = self.width
            hdr[1] = self.height
            t = self._d.dtype
            if t == np.uint8 or t == np.int8:
                hdr[2] = 0
            elif t == np.uint16 or t == np.int16:
                hdr[2] = 1
            elif t == np.uint32 or t == np.int32:
                hdr[2] = 2
            elif t == np.float64:
                hdr[2] = 3
            else:
                return False
            fl.write(hdr.tobytes())
            fl.write(self._d.tobytes())
            fl.close()
            return True
        return False

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __len__(self):
        return len(self._d)

    def __bytes__(self):
        return bytes(self._d)

    def __lt__(self, c):
        if isinstance(c, FLDMap):
            idx = self._d < c._d
        else:
            idx = self._d < float(c)
        return idx

    def __gt__(self, c):
        if isinstance(c, FLDMap):
            idx = self._d > c._d
        else:
            idx = self._d > float(c)
        return idx

    def __eq__(self, c):
        if isinstance(c, FLDMap):
            idx = self._d == c._d
        else:
            idx = self._d == float(c)
        return idx

    def max(self, only_value=True):
        """ Find pixel containing maximum value.
        """
        if only_value: return self._d.max()
        yMax = 0
        xMax = 0
        maxVal = None
        for y in range(self._d.shape[0]):
            for x in range(self._d.shape[1]):
                val = self._d[y, x]
                if maxVal is None or val > maxVal:
                    yMax = y
                    xMax = x
                    maxVal = val
        return (xMax, yMax, maxVal)

    def min(self, only_value=True):
        """ Find pixel containing minimum value.
        """
        if only_value: return self._d.min()
        yMin = 0
        xMin = 0
        minVal = None
        for y in range(self._d.shape[0]):
            for x in range(self._d.shape[1]):
                val = self._d[y, x]
                if minVal is None or val < minVal:
                    yMin = y
                    xMin = x
                    minVal = val
        return (xMin, yMin, minVal)

    def __iadd__(self, c):
        if isinstance(c, FLDMap):
            self._d = self._d + c._d
        else:
            self._d = self._d + float(c)
        return self

    def __isub__(self, c):
        if isinstance(c, FLDMap):
            self._d = self._d - c._d
        else:
            self._d = self._d - float(c)
        return self

    def __imul__(self, c):
        if isinstance(c, FLDMap):
            self._d = self._d * c._d
        else:
            self._d = self._d * float(c)
        return self

    def __itruediv__(self, c):
        if isinstance(c, FLDMap):
            self._d = self._d / c._d
        else:
            self._d = self._d / float(c)
        return self

    def __add__(self, c):
        r = FLDMap(self)
        r += c
        return r

    def __sub__(self, c):
        r = FLDMap(self)
        r -= c
        return r

    def __mul__(self, c):
        r = FLDMap(self)
        r *= c
        return r

    def __str__(self):
        s = f'FLDMap {self.width}x{self.height} object'
        return s

    def __truediv__(self, c):
        r = FLDMap(self)
        r /= c
        return r

    def mean(self, r):
        if len(r) != 4: raise ValueError('argument should be of length 4')
        ce = r[2]
        re = r[3]
        if ce < 0: ce += self.width
        if re < 0: re += self.height
        ce += 1
        re += 1
        n = (ce - r[0])*(re - r[1])
        return self._d[r[1]:re, r[0]:ce].sum() / n

    def std(self, r):
        if len(r) != 4: raise ValueError('argument should be of length 4')
        if r[2] < 0: r[2] += self.width
        if r[3] < 0: r[3] += self.height
        r[2] += 1
        r[3] += 1
        return self._d[r[0]:r[2], r[1]:r[3]].std()

    def reciprocal(self):
        r = FLDMap()
        r._d = 1 / self._d
        return r

    def remove(self, rows=None, columns=None):
        if rows: self._d = np.delete(self._d, rows, 0)
        if columns: self._d = np.delete(self._d, columns, 1)

    @classmethod
    def random(cls, w, h, loc=0.0, scale=1.0, dtype=np.float64, method='uniform'):
        b = cls()
        if method == 'gauss' or method == 'normal':
            b._d = np.random.normal(loc, scale, [h, w])
        elif method == 'lognormal':
            b._d = np.random.lognormal(loc, scale, [h, w])
        else:
            b._d = np.random.rand(h, w)*scale + loc
        if dtype != np.float64:
            b._d = b._d.astype(dtype)
        return b

    @classmethod
    def pattern(cls, w, h, dtype=np.uint16):
        b = cls()
        b._d = np.ndarray([h, w], dtype)
        for y in range(h):
            for x in range(w):
                b._d[y][x] = x + y
        np.clip(b._d, 0, w - 1, b._d)
        return b

    @property
    def width(self): return self._d.shape[1]

    @property
    def height(self): return self._d.shape[0]

class Pgm(object):
    """ Pgm class
    """
    Types = [np.uint8, np.uint16, np.uint32, np.float64]

    def __init__(self, w=0, h=0, *, dtype=np.ushort, maxval=65535, _d=None):
        if isinstance(w, FLDMap):
            self._flags = ()
            self._pars = {}
            self._d = w._d
            self._maxval = np.nanmax(self._d)
        elif isinstance(w, Pgm):
            self._flags = w._flags
            self._pars = w._pars
            self._d = np.copy(w._d)
            self._maxval = w._maxval
        else:
            self._flags = ()
            self._pars = {}
            if dtype not in self.Types:
                dtype = np.ushort
            if isinstance(_d, np.ndarray):
                self._d = np.reshape(_d.astype(dtype), (h, w))
            elif isinstance(_d, bytes):
                self._d = np.reshape(np.frombuffer(_d, dtype), (h, w))
            else:
                self._d = np.zeros((h, w), dtype)
            if dtype == np.float64:
                maxval = np.float64(maxval)
            else:
                maxval = min(maxval, np.iinfo(dtype).max)
                maxval = max(maxval, np.iinfo(dtype).min)
            self._maxval = maxval

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __len__(self):
        return len(self._d)

    def __bytes__(self):
        return bytes(self._d)

    def load(self, filepath: str) -> bool:
        """ Load pgm object from file.
        """
        ln = b''
        try:
            file = open(filepath, 'rb')
        except FileNotFoundError as e:
            print(str(e))
            return False

        def readln():
            nonlocal ln
            ln = file.readline().rstrip().lstrip()
            return ln is not None
        if file.readline().rstrip().lstrip() != b'P5':
            file.close()
            return False
        pars = {}
        flags = ()
        while readln() and ln[0] == 35:
            x = ln.split(b'=')
            if len(x) == 2:
                key = x[0][1:].decode().rstrip()
                val = x[1]
                dt = getdatetime(val)
                if dt: pars[key] = dt; continue
                try: val = int(val)
                except ValueError:
                    try: val = float(val)
                    except ValueError:
                        val = val.decode('utf-8')
                pars[key] = val
            else:
                flags += (ln[1:].decode(),)
        x = ln.split(b' ')
        try:
            x = [int(x[0]), int(x[1])]
        except ValueError:
            file.close()
            return False
        maxval = 0
        try:
            ln = file.readline().rstrip().lstrip()
            maxval = int(ln)
        except ValueError:
            file.close()
            return False
        nb = 2
        dtype = np.uint16
        if maxval < 256:
            nb = 1
            dtype = np.uint8
        elif maxval > 65535:
            nb = 4
            dtype = np.uint32
        b = file.read(x[0]*x[1]*nb)
        if not file.read():
            self._pars = pars
            self._flags = flags
            self._d = np.frombuffer(b, dtype).reshape(x[1], x[0])
            if 'Shift' in pars:
                b = 1 / (1 << pars['Shift'])
                self._d = self._d * b
                del pars['Shift']
            self._maxval = self._d.max()
        file.close()
        return True

    def save(self, filepath: str) -> bool:
        """ Save pgm object to file.
        """
        file = open(filepath, 'wb')
        if file is None:
            return False
        file.write(b'P5\n')
        s = '\n'.join([f'#{m}' for m in self._flags]) + '\n'
        file.write(str.encode(s))
        s = '\n'.join([f'#{m} = {n}' for m, n in self._pars.items()]) + '\n'
        file.write(str.encode(s))
        maxval = self._d.max()
        sh = 0
        if self._d.dtype == np.float64:
            while maxval < (1 << 30) and sh < 31:
                maxval *= 2
                sh += 1
            while maxval >= (1 << 31) and sh > -31:
                maxval *= 0.5
                sh -= 1
            s = f'#Shift = {sh}\n'
            file.write(str.encode(s))
            maxval = np.uint32(maxval)
        s = f'{self._d.shape[1]} {self._d.shape[0]}\n{maxval}\n'
        file.write(str.encode(s))
        if sh == 0:
            file.write(self._d.tobytes())
        else:
            sh = (1 << sh) if sh > 0 else (1/(1 << (-sh)))
            file.write(np.uint32(self._d * sh).tobytes())
        file.close()
        return True

    def exportpng(self, path: str, vmin=0.0, vmax=np.nan):
        ext = path[-4:].lower()
        if ext != '.png':
            path += '.png'
        if np.isnan(vmax):
            vmax = self._maxval
        if self._d.dtype == np.float:
            d = self._d
        else:
            d = self._d.astype(np.float)
        vmax = 255.0 / (vmax - vmin)
        d = (d - vmin) * vmax
        d = d.astype(np.uint8)
        if cv2 is not None:
            cv2.imwrite(path, d)

    def show(self, name='PGM'):
        """ Pgm object visualization.
        """
        d = self._d / self._maxval
        if cv2:
            #cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, d)
            cv2.waitKey(0)
            cv2.destroyWindow(name)

    def fill(self, val=0, *, pattern='None'):
        """ Fills data with given pattern.
            Possible patterns:
                Random - random values from 0 to val
                None or any other name - fills all cells with the given value
        """
        if pattern == 'Random':
            d = np.random.rand(self._d.shape[0], self._d.shape[1])
            self._d = (d*val).astype(self._d.dtype)
        else:
            self._d[:] = val

    def __lt__(self, c):
        if isinstance(c, Pgm):
            idx = self._d < c._d
        else:
            c = float(c)
            idx = self._d < c
        return idx

    def __gt__(self, c):
        if isinstance(c, Pgm):
            idx = self._d > c._d
        else:
            c = float(c)
            idx = self._d > c
        return idx

    def __eq__(self, c):
        if isinstance(c, Pgm):
            idx = self._d == c._d
        else:
            c = float(c)
            idx = self._d == c
        return idx

    def max(self):
        """ Find pixel containing maximum value.
        """
        yMax = 0
        xMax = 0
        maxVal = None
        for y in range(self._d.shape[0]):
            for x in range(self._d.shape[1]):
                val = self._d[y, x]
                if maxVal is None or val > maxVal:
                    yMax = y
                    xMax = x
                    maxVal = val
        return (xMax, yMax, maxVal)

    def min(self):
        """ Find pixel containing minimum value.
        """
        yMin = 0
        xMin = 0
        minVal = None
        for y in range(self._d.shape[0]):
            for x in range(self._d.shape[1]):
                val = self._d[y, x]
                if minVal is None or val < minVal:
                    yMin = y
                    xMin = x
                    minVal = val
        return (xMin, yMin, minVal)

    def getquantiles(self, ql):
        if isinstance(ql, np.ndarray):
            if ql.dtype != float:
                ql = ql.astype(float)
        elif isinstance(ql, list):
            for k, v in enumerate(ql):
                ql[k] = float(v)
        else: ql = [float(ql)]

        d = self._d.flatten()
        n = len(d)
        idx = heapsort(d, verbose=True)
        qrv = []
        _w = self.width
        for q in ql:
            try:
                k = idx[int(q*n)]
                y = k // _w
                x = k - y * _w
                qrv.append((x, y, d[k]))
            except IndexError:
                qrv.append((None, None, None))
        return qrv

    def getphc(self, xleft, ytop, xright, ybottom):
        """ Function calculates in the given rectangle
            coordinates, signal and signal-to-noise ratio
            of a star-like image (photocenter).
            It uses 1 pixel perimeter around the rectangle
            to do noise estinamation.
        """
        fSum = np.int64(0)
        xSum = np.int64(0)
        ySum = np.int64(0)
        nSum = np.int64(0)
        bSum = np.int64(0)
        fCnt = np.int64(0)
        bCnt = np.int64(0)
        if xleft < 0:
            xleft = 0
        if xright >= self._d.shape[1]:
            xright = self._d.shape[1] - 1
        if ytop < 0:
            ytop = 0
        if ybottom >= self._d.shape[0]:
            ybottom = self._d.shape[0] - 1
        for y in range(ytop, ybottom + 1):
            bck = (y == ytop) or (y == ybottom)
            for x in range(xleft, xright + 1):
                v = np.int64(self._d[y, x])
                if bck or (x == xleft) or (x == xright):
                    bCnt += 1
                    bSum += v
                    nSum += v*v
                else:
                    fCnt += 1
                    fSum += v
                    xSum += x*v
                    ySum += y*v
        nSum = (nSum - bSum*bSum/bCnt) / (bCnt - 1)
        bSum = bSum / bCnt
        xSum = xSum / fSum
        ySum = ySum / fSum
        fSum = fSum - bSum*fCnt
        return xSum, ySum, fSum, fSum / np.sqrt(nSum*fCnt)

    def findphc(self, size: int):
        """ Search for single photocenter.
        """
        p = self.max()
        return self.getphc(p[0] - size, p[1] - size, p[0] + size, p[1] + size)

    def _checkarg(self, c):
        if not isinstance(c, self._d.dtype.type):
            c = np.float64(c)
            if self._d.dtype != np.float64:
                self._d = self._d.astype(np.float64)
        return c

    def __iadd__(self, c):
        if isinstance(c, Pgm):
            self._d = self._d + c._d
        else:
            c = self._checkarg(c)
            self._d = self._d + c
        self._maxval = self._d.max()
        return self

    def __isub__(self, c):
        if isinstance(c, Pgm):
            self._d = self._d - c._d
        else:
            c = self._checkarg(c)
            self._d = self._d - c
        self._maxval = self._d.max()
        return self

    def __imul__(self, c):
        if isinstance(c, Pgm):
            self._d = self._d * c._d
        else:
            c = self._checkarg(c)
            self._d = self._d * c
        self._maxval = self._d.max()
        return self

    def __itruediv__(self, c):
        if isinstance(c, Pgm):
            self._d = self._d / c._d
        else:
            c = self._checkarg(c)
            self._d = self._d / c
        self._maxval = self._d.max()
        return self

    def sqr(self):
        d = np.float64(self._maxval)
        d = d*d
        if self._d.dtype != np.float64:
            if d > np.float64(np.iinfo(self._d.dtype).max):
                self._d = self._d.astype(np.float64)
                self._maxval = d
            else:
                self._maxval *= self._maxval
        self._d = self._d * self._d
        return self

    def __add__(self, c):
        r = Pgm(self)
        r += c
        return r

    def __sub__(self, c):
        r = Pgm(self)
        r -= c
        return r

    def __mul__(self, c):
        r = Pgm(self)
        r *= c
        return r

    def __str__(self):
        s = f'Pgm object: {self.width}x{self.height} pixels of type {self._d.dtype}'
        return s

    def __truediv__(self, c):
        r = Pgm(self)
        r /= c
        return r

    def subframe(self, xl, yt, xr, yb):
        if xl < 0:
            xl = 0
        if xr >= self.width:
            xr = self.width
        else:
            xr += 1
        if yt < 0:
            yt = 0
        if yb >= self.height:
            yb = self.height
        else:
            yb += 1
        if xl >= xr or yt >= yb:
            return None
        pr = Pgm(xr - xl, yb - yt, dtype=self._d.dtype)
        pr._maxval = 0
        for y in range(0, pr.height):
            pr._d[y, :] = self._d[y + yt, xl:xr]
        return pr

    def pvalue(self, name):
        if any(name in k for k in self._flags):
            return True
        if name in self._pars:
            return self._pars[name]
        return None

    def imagedata(self):
        d = self._d / self._maxval * 255
        d = d.astype(np.uint8)
        return d

    def mean(self, r):
        if len(r) != 4:
            raise ValueError('argument should be of length 4')
        if r[2] < 0: r[2] += self.width
        if r[3] < 0: r[3] += self.height
        r[2] += 1
        r[3] += 1
        n = (r[2] - r[0])*(r[3] - r[1])
        return self._d[r[0]:r[2], r[1]:r[3]].sum() / n

    @classmethod
    def sqrt(cls, c):
        r = cls(c)
        if r._d.dtype != np.float64:
            r._d = r._d.astype(np.float64)
        r._d = np.sqrt(r._d)
        return r

    @property
    def width(self): return self._d.shape[1]

    @property
    def height(self): return self._d.shape[0]

    @property
    def shape(self): return self._d.shape

    @property
    def isempty(self): return self._d.shape[0] == 0 and self._d.shape[1] == 0

    @property
    def data(self): return self._d

    @classmethod
    def fromfile(cls, filepath: str):
        """ Load pgm object from file.
        """
        a = cls()
        a.load(filepath)
        return a

    @classmethod
    def fromdata(cls, d: np.ndarray, *, dtype=None):
        """ Create pgm object from numpy 2D array
        """
        if not isinstance(d, np.ndarray) or d.ndim != 2:
            raise ValueError('Argument should be 2D numpy array')
        a = cls(d.shape[1], d.shape[0])
        a._maxval = d.max()
        if dtype in cls.Types:
            a._d = dtype(d)
            a._maxval = dtype(a._maxval)
        else:
            a._d = d
        return a

    @classmethod
    def average(cls, dpath):
        files = glob(dpath + "/*.pgm")
        n = len(files)
        if n == 0:
            return (None, None)
        p = cls.fromfile(files[0])
        pAvg = cls(p.width, p.height, dtype=np.float64)
        pStDev = cls(p.width, p.height, dtype=np.float64)
        pbar = tqdm(total=n) if tqdm else None
        p0 = None
        for fl in files:
            p = cls.fromfile(fl)
            pAvg += p
            pStDev += p.sqr()
            if pbar: pbar.update()
            p0 = p0 or p
        pStDev = Pgm.sqrt((pStDev - (pAvg*pAvg) / n) / (n - 1))
        pAvg /= n
        pStDev._pars = p0._pars
        pStDev._flags = p0._flags
        pAvg._pars = p0._pars
        pAvg._flags = p0._flags
        return (pAvg, pStDev)

def createvideo(dirpath: str, vidpath=None, filefilter=None, show=False):
    if cv2 is None:
        print("Error: CV2 module is missing")
        return
    if dirpath.endswith('/'):
        dirpath = dirpath[:-1]
    if not isinstance(filefilter, str):
        filefilter = '*.pgm'
    files = glob(dirpath + '/' + filefilter)
    if len(files) == 0:
        return
    if not isinstance(vidpath, str):
        vidpath = dirpath + '.avi'

    if tqdm is not None:
        pbar = tqdm(total=len(files))

    p = Pgm.fromfile(files[0])
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(vidpath, 0, 20, (p.width, p.height), False)

    for fl in files:
        p = Pgm.fromfile(fl)
        video.write(p.imagedata())
        if tqdm is not None: pbar.update()
        if show: p.show()
    video.release()

if __name__ == "__main__":
    pp = Pgm.fromfile('d:/Users/Simae/Work/data/2020.04.02/bias.pgm')
    ql = pp.getquantiles([0.1, 0.5, 0.9])
    print(*ql)
    #pp.savepng('d:/Users/Simae/Work/data/2020.04.02/bias.png')
    pp.show()
    #createvideo('d:/Users/Simae/Work/data/2020.03.26/5')
