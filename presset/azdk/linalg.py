""" Quaternion & Vector class implementations
"""

from __future__ import annotations
import re
from math import isclose
import numpy as np

class Vector:
    """ Vector class
    """
    def __init__(self, *w):
        if len(w) == 0:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
        elif len(w) == 1:
            if isinstance(w[0], Vector):
                self.x = w[0].x
                self.y = w[0].y
                self.z = w[0].z
            elif isinstance(w[0], float):
                self.x = w[0]
                self.y = w[0]
                self.z = w[0]
            elif len(w[0]) == 3:
                self.x = float(w[0][0])
                self.y = float(w[0][1])
                self.z = float(w[0][2])
            else:
                raise ValueError('Single argument should be Vector or float')
        elif len(w) == 3:
            self.x = float(w[0])
            self.y = float(w[1])
            self.z = float(w[2])
        else:
            raise ValueError('Number of arguments should be 0, 1 or 3')

    def __bool__(self):
        return self.x != 0.0 or self.y != 0.0 or self.z != 0.0

    def __iadd__(self, c):
        if isinstance(c, Vector):
            self.x += c.x
            self.y += c.y
            self.z += c.z
        else:
            c = float(c)
            self.x += c
            self.y += c
            self.z += c
        return self

    def __isub__(self, c):
        if isinstance(c, Vector):
            self.x -= c.x
            self.y -= c.y
            self.z -= c.z
        else:
            c = float(c)
            self.x -= c
            self.y -= c
            self.z -= c
        return self

    def __imul__(self, c):
        if isinstance(c, Vector):
            self.x *= c.x
            self.y *= c.y
            self.z *= c.z
        else:
            c = float(c)
            self.x *= c
            self.y *= c
            self.z *= c
        return self

    def __itruediv__(self, c):
        if isinstance(c, Vector):
            self.x /= c.x
            self.y /= c.y
            self.z /= c.z
        else:
            c = float(c)
            self.x /= c
            self.y /= c
            self.z /= c
        return self

    def __add__(self, c):
        b = Vector(self)
        b += c
        return b

    def __sub__(self, c):
        b = Vector(self)
        b -= c
        return b

    def __mul__(self, c):
        b = Vector(self)
        b *= c
        return b

    def __truediv__(self, c):
        b = Vector(self)
        b /= c
        return b

    def __str__(self):
        return f'{{{self.x:10.6f}, {self.y:10.6f}, {self.z:10.6f}}}'

    def __eq__(self, b : Vector):
        return isclose(self.x, b.x) and isclose(self.y, b.y) and isclose(self.z, b.z)

    def cross(self, c):
        return Vector(
            self.y*c.z - self.z*c.y,
            self.z*c.x - self.x*c.z,
            self.x*c.y - self.y*c.x
        )

    def data(self):
        return np.array([self.x, self.y, self.z])

    def dot(self, c):
        return self.x*c.x + self.y*c.y + self.z*c.z

    def normalize(self):
        l = self.length
        if l > 1e-6: self /= l
        return l

    def normalized(self):
        b = Vector(self)
        b.normalize()
        return b

    def isclose(self, v, *e):
        e = Vector(*e)
        if isinstance(v, Vector):
            if np.abs(self.x - v.x) > e.x: return False
            if np.abs(self.y - v.y) > e.y: return False
            if np.abs(self.z - v.z) > e.z: return False
            return True
        return False

    def tobytes(self):
        return self.data().tobytes()

    def toangles(self, coef=1.0):
        ra = np.arctan2(self.y, self.x)
        de = np.arcsin(self.z / self.length)
        return (ra*coef, de*coef)

    @property
    def length(self) -> float:
        return np.sqrt(self.dot(self))

    @property
    def norm(self):
        return self.dot(self)

    @classmethod
    def fromstr(cls, vs: str):
        if isinstance(vs, bytes):
            vs = vs.decode()
        reDbl = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        res = re.findall(reDbl, vs)
        v = cls()
        if len(res) == 3:
            v.x = float(res[0])
            v.y = float(res[1])
            v.z = float(res[2])
            return v
        else: return None

    @classmethod
    def randomdir(cls, length=1.0):
        ds = 0.0
        while ds < 0.01 or ds > 1.0:
            d = np.random.random(3)*2.0 - 1.0
            ds = sum(d**2)
        d = d * (length / np.sqrt(ds))
        return cls(*d)

    @classmethod
    def frombuffer(cls, d: bytes):
        if len(d) != 32:
            return cls()
        d = np.frombuffer(d, np.float, 4)
        return cls(*d)

    @classmethod
    def fromangles(cls, ra : float, de : float):
        v = cls()
        v.x = np.cos(ra) * np.cos(de)
        v.y = np.sin(ra) * np.cos(de)
        v.z = np.sin(de)
        return v

class Quaternion:
    """ Quaternion class
    """
    def __init__(self, *w):
        if len(w) == 4:
            self.w = float(w[0])
            self.x = float(w[1])
            self.y = float(w[2])
            self.z = float(w[3])
        elif len(w) == 2:
            if isinstance(w[1], float) and isinstance(w[0], Vector):
                sx = np.sin(w[1] * 0.5) / w[0].length
                self.w = np.cos(w[1]*0.5)
                self.x = sx * w[0].x
                self.y = sx * w[0].y
                self.z = sx * w[0].z
            else:
                raise ValueError('Pair arguments should be Vector and float')
        elif len(w) == 1:
            if isinstance(w[0], float):
                self.w = w[0]
                self.x = w[0]
                self.y = w[0]
                self.z = w[0]
            elif isinstance(w[0], Quaternion):
                self.w = w[0].w
                self.x = w[0].x
                self.y = w[0].y
                self.z = w[0].z
            else:
                raise ValueError('Single argument should be Quaternion or float')
        elif len(w) == 0:
            self.w = 0.0
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
        else:
            raise ValueError('Number of arguments should be 0, 1, 2 or 4')
        #if self.w < 0: self.inverse()

    def __bool__(self):
        return self.w != 0.0 or self.x != 0.0 or self.y != 0.0 or self.z != 0.0

    def __iadd__(self, c):
        if isinstance(c, Quaternion):
            self.w += c.w
            self.x += c.x
            self.y += c.y
            self.z += c.z
        else:
            c = float(c)
            self.w += c
            self.x += c
            self.y += c
            self.z += c
        return self

    def __isub__(self, c):
        if isinstance(c, Quaternion):
            self.w -= c.w
            self.x -= c.x
            self.y -= c.y
            self.z -= c.z
        else:
            c = float(c)
            self.w -= c
            self.x -= c
            self.y -= c
            self.z -= c
        return self

    def __imul__(self, c):
        b = self * c
        self.w = b.w
        self.x = b.x
        self.y = b.y
        self.z = b.z
        return self

    def __add__(self, c):
        b = Quaternion(self)
        b += c
        return b

    def __sub__(self, c):
        b = Quaternion(self)
        b -= c
        return b

    def __mul__(self, c):
        if isinstance(c, Quaternion):
            w = self.w*c.w - self.x*c.x - self.y*c.y - self.z*c.z
            x = self.w*c.x + self.x*c.w + self.y*c.z - self.z*c.y
            y = self.w*c.y - self.x*c.z + self.y*c.w + self.z*c.x
            z = self.w*c.z + self.x*c.y - self.y*c.x + self.z*c.w
        elif isinstance(c, Vector):
            w = -(self.x*c.x + self.y*c.y + self.z*c.z)
            x = c.w*self.x + self.y*c.z - self.z*c.y
            y = c.w*self.y + self.z*c.x - self.x*c.z
            z = c.w*self.z + self.x*c.y - self.y*c.x
        else:
            c = float(c)
            w = self.w * c
            x = self.x * c
            y = self.y * c
            z = self.z * c
        return Quaternion(w, x, y, z)

    def __truediv__(self, c):
        if isinstance(c, float):
            self.w /= c
            self.x /= c
            self.y /= c
            self.z /= c
        else:
            raise ValueError
        return self

    def __str__(self):
        return f'{{{self.w:9.6f}, {self.x:9.6f}, {self.y:9.6f}, {self.z:9.6f}}}'

    def __eq__(self, q):
        if isinstance(q, Quaternion):
            if np.abs(self.w - q.w) > 1e-12: return False
            if np.abs(self.x - q.x) > 1e-12: return False
            if np.abs(self.y - q.y) > 1e-12: return False
            if np.abs(self.z - q.z) > 1e-12: return False
            return True
        return False

    def __pow__(self, n):
        pn = float(n)
        qn = Quaternion(self)
        qp = np.sqrt(qn.norm)
        qn /= qp
        qv, qa = qn.decompose()
        qp = qp**pn
        qn = Quaternion(qv, qa*pn) * qp
        return qn

    def conjugate(self):
        coef = 1.0 / np.sqrt(self.norm)
        self.x = -self.x*coef
        self.y = -self.y*coef
        self.z = -self.z*coef
        return self

    def conjugated(self):
        q = Quaternion(self)
        return q.conjugate()

    def inverse(self):
        self.w = -self.w
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def rotate(self, v: Vector):
        if not isinstance(v, Vector):
            raise ValueError()
        n = self.norm
        if n == 0.0:
            return Vector()
        if np.abs(n - 1.0) > 1e-12:
            n = np.sqrt(n)
        elif np.abs(self.w - 1.0) < 1e-12:
            return Vector(v)
        qv = Quaternion(0, v.x, v.y, v.z)
        qv = qv * Quaternion(self.w/n, -self.x/n, -self.y/n, -self.z/n)
        qv = self * qv
        return Vector(qv.x, qv.y, qv.z)

    def rotateInverse(self, v: Vector):
        if not isinstance(v, Vector):
            raise ValueError()
        n = self.norm
        if n == 0.0:
            return Vector()
        if np.abs(n - 1.0) > 1e-12:
            n = np.sqrt(n)
        elif np.abs(self.w - 1.0) < 1e-12:
            return Vector(v)
        qv = Quaternion(self.w/n, -self.x/n, -self.y/n, -self.z/n)
        qv = qv * Quaternion(0, v.x, v.y, v.z)
        qv = qv * self
        return Vector(qv.x, qv.y, qv.z)

    def toangles(self, zeropass=True):
        if self.isnull:
            return Vector(np.nan, np.nan, np.nan)
        nz = self * Quaternion(self.z, self.y, -self.x, self.w)
        ra = np.arctan2(nz.y, nz.x)
        if zeropass and ra < 0: ra += 2 * np.pi
        de = np.arcsin(nz.z)
        qq = Quaternion(0.5)
        qq *= Quaternion(self.w, -self.x, -self.y, -self.z)
        qq *= Quaternion(Vector(0, 0, 1), ra)
        qq *= Quaternion(Vector(0, 1, 0), -de)
        phi = -np.arctan2(qq.x, qq.w) * 2.0
        if phi < -np.pi/8: phi += 2 * np.pi
        elif phi > np.pi*7/8: phi -= 2 * np.pi
        return Vector(ra, de, phi)

    def data(self):
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        n = self.norm
        if np.abs(n - 1) < 1e-12:
            return self
        if n == 0.0:
            self.x = self.y = self.z = self.w = np.nan
        else:
            self *= 1.0 / np.sqrt(n)
        return self

    def decompose(self):
        """ Decompose normalized quaternion to vector + angles pair
        """
        if not self.isunit:
            return Vector(np.nan), np.nan
        ang = float(np.arccos(self.w)) * 2.0
        sa = np.sin(ang * 0.5)
        if np.abs(sa) < 1e-6:
            v = Vector(1, 0, 0)
        else:
            v = Vector(self.x / sa, self.y / sa, self.z / sa)
        return v, ang

    def dot(self, q):
        if not isinstance(q, Quaternion):
            raise ValueError('Argument should be Quaternion')
        dqa = self.w * q.w + self.x * q.x + self.y * q.y + self.z * q.z
        #dqa = np.arccos(dqa)
        return dqa

    def tobytes(self):
        return self.data().tobytes()

    @property
    def norm(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def isnull(self):
        return isclose(self.norm, 0)

    @property
    def isunit(self):
        return isclose(self.norm, 1.0)

    @property
    def isidentity(self):
        return self.isunit and isclose(self.w, 1.0)

    @property
    def isnan(self):
        return np.isnan(self.w) or np.isnan(self.x) or np.isnan(self.y) or np.isnan(self.z)

    @classmethod
    def fromstr(cls, qs: str):
        if isinstance(qs, bytes):
            qs = qs.decode()
        reDbl = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        res = re.findall(reDbl, qs)
        q = cls()
        if len(res) == 4:
            q.w = float(res[0])
            q.x = float(res[1])
            q.y = float(res[2])
            q.z = float(res[3])
            return q
        else: return None

    @classmethod
    def fromangles(cls, angs, coef = 1.0):
        if isinstance(angs, Vector):
            ra = angs.x*coef
            de = angs.y*coef
            fi = angs.z*coef
        elif (isinstance(angs, np.ndarray) or isinstance(angs, list)) and len(angs) == 3:
            ra = float(angs[0])*coef
            de = float(angs[1])*coef
            fi = float(angs[2])*coef
        else:
            return Quaternion(np.nan)
        q = cls(np.cos(ra*0.5), 0, 0, np.sin(ra*0.5))
        q *= cls(np.cos(de*0.5), 0, -np.sin(de*0.5), 0)
        q *= cls(np.cos(fi*0.5), np.sin(fi*0.5), 0, 0)
        q *= cls(0.5)
        return q

    @classmethod
    def random(cls):
        ds = 0.0
        while ds < 0.01 or ds > 1.0:
            d = np.random.random(4)*2.0 - 1.0
            ds = sum(d**2)
        d = d / np.sqrt(ds)
        return cls(*d)

    @classmethod
    def frombuffer(cls, d: bytes):
        if len(d) != 32:
            return cls()
        d = np.frombuffer(d, np.float, 4)
        return cls(*d)

def vslerp(va: Vector, vb: Vector, t: float):
    v = va.cross(vb)
    vl = v.length
    if np.abs(vl) < 1e-6: return Vector(va)
    st = v / vl
    theta = np.arcsin(st)
    v = va*(np.sin(theta*(1-t))/st) + vb*(np.sin(theta*t)/st)
    return v

def qslerp(qa: Quaternion, qb: Quaternion, t: float):
    q = Quaternion(qa)
    q = qb*q.conjugate()
    return (q**t)*qa

class Rotator:
    def __init__(self, _vspd : Vector = None, _qs : Quaternion = None):
        if isinstance(_vspd, Rotator):
            self.dir = Vector(_vspd.dir)
            self.speed = _vspd.speed
            self.qs = Quaternion(_vspd.qs)
        else:
            if isinstance(_vspd, Vector):
                self.dir = _vspd
                self.speed = self.dir.normalize()
                if self.speed < 1e-6:
                    self.dir = Vector(0, 0, 1)
            else:
                self.dir = Vector(0, 0, 1)
                self.speed = 0.0
            if isinstance(_qs, Quaternion):
                self.qs = Quaternion(_qs)
                self.qs.normalize()
            else:
                self.qs = Quaternion(1, 0, 0, 0)

    def __str__(self):
        s = 'Qz=' + str(self.qs)
        s += f', Vr={self.speed:4.2f}*' + str(self.dir)
        return s

    def rotation(self, t: float) -> Quaternion:
        q = Quaternion(self.dir, self.speed * t)
        q *= self.qs
        if q.w < 0: q.inverse()
        return q

    def setquat(self, _qs: Quaternion, _ts = 0.0):
        self.qs = Quaternion(_qs)
        self.qs.normalize()

    def setvelocity(self, _vs: Vector, _ts = 0.0):
        q = self.rotation(_ts)
        self.dir = Vector(_vs)
        self.speed = self.dir.normalize()
        self.qs = q

    @classmethod
    def fromquat(cls, qa: Quaternion, qb: Quaternion, tDelta: float):
        qr = Quaternion(qa)
        qr = qb*qr.conjugate()
        qaxis, qangle = qr.decompose()
        return cls(qaxis*qangle / tDelta, qa)

class Summator:
    def __init__(self):
        self._quats = []
        self._times = []

    def append(self, q: Quaternion, t: float):
        if isinstance(q, Quaternion):
            q = [q]
            t = [t]
        if isinstance(q, list):
            n = len(q)
            if len(t) != n or n == 0:
                ValueError('Lengths of input lists do not match')
            for k in range(n):
                if isinstance(q[k], Quaternion):
                    self._quats.append(q[k])
                    self._times.append(float(t[k]))
        return len(self._quats)

if __name__ == "__main__":
    q = Quaternion(0.037076,  0.037108,  0.706180,  0.706086).normalize()
    q.toangles()
    _qs = ' { 0.493789,  0.511382,  0.491488,  0.503253}'
    _vs = ' { 0.5,  0.5,  0.5}'
    #_q = Quaternion.fromstr(_qs)
    np.random.seed(1234567)
    q1 = Quaternion.random()
    q2 = Quaternion.random()
    qr = Rotator.fromquat(q1, q2, 1.0)
    print(q1, q2)
    n = 30
    vx = Vector(1, 0, 0)
    for ti in range(n+1):
        t = ti / n
        q = qslerp(q1, q2, t)
        vq = q.rotate(vx)
        qb = qr.rotation(t)
        vqb = qb.rotate(vx)
        da = vq.dot(vqb)
        da = np.arccos(da)*206265 if da < 1 else 0
        print(f'{t:5.3f}', q, qb, f'{da:.0f}')
