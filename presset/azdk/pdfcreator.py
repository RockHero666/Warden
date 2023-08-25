import os
import sys
import time
import datetime as dt
from enum import Enum
from glob import glob
import numpy as np
from pandas import read_csv, DataFrame
from fpdf import FPDF, HTMLMixin
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from azdk.utils import AzdkThread, RegExp, to_hm_str


def checkTimesOverMidnight(times : list):
    t0 = times[0]
    times = [t if t >= t0 else t + dt.timedelta(days=1) for t in times]
    return times

class Date:
    months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
    def __init__(self, _date : dt.date):
        self.d = _date.day
        self.m = _date.month
        self.y = _date.year

    def __str__(self):
        m = self.months[self.m-1]
        return f'{self.d:02d} {m} {self.y:04d} г.'

    def __eq__(self, c):
        if c is None: return False
        return self.d == c.d and self.m == c.m and self.y == c.y

class Pdf(FPDF, HTMLMixin):
    class Section():
        def __init__(self, name : str):
            self.name = name
            self.subsections = []
            self.iterator = 0

        def append(self, name : str):
            s = self.__class__(name)
            self.subsections.append(s)

        def __iter__(self):
            self.iterator = 0
            return self

        def __next__(self):
            k = self.iterator + 1
            if k > len(self.subsections):
                raise StopIteration
            self.iterator = k
            return self.subsections[k-1]

        def __contains__(self, name : str):
            return name in self.subsections

        def __getitem__(self, idx : int):
            return self.subsections[idx]

    def __init__(self, title : str = None, margins = (15, 10, 15, 10),
                 author=None, verbose=False, downscale_images=False):
        super().__init__("portrait", "mm", "A4")
        self.sections = []
        if isinstance(title, str) and len(title) > 0:
            self.set_title(title)
        self.set_lang('Русский')
        if isinstance(author, str): self.set_author(author)
        self.set_margins(*margins[:3])
        self.set_auto_page_break(True, margins[3])
        self.fig_counter = 0
        self.table_counter = 0
        self.log_file = None
        self.verbose = verbose
        self.oversized_images = 'downscale' if downscale_images else None
        self.changed = False
        self.inline_checking = False
        self.disable_first_page_number = True
        self.print_sec_num = True
        self.print_timestamp = False
        self.logger = None
        self.organization = None
        self.add_font('FreeSerif', '', 'azdk/fonts/FreeSerif.ttf')
        self.add_font('FreeSerif', 'B', 'azdk/fonts/FreeSerifBold.ttf')

    def _log(self, txt : str):
        if not self.verbose: return
        if self.print_timestamp:
            txt = dt.datetime.now().strftime('%H:%M:%S: ') + txt
        print(txt, file=self.logger)

    def _settitlepage(self, title : str):
        self.add_page('P')
        self.set_font('Times', 'B', 16)
        self.cell(210/2, 297/2, title, align='C')

    def _check_sup_sub(self, txt : str, fontsz : int):
        for m in RegExp.reSupSub.findall(txt):
            if len(m[0]) > 0: self.write(fontsz/2, m[0])
            match m[2]:
                case '^':
                    self.char_vpos = 'SUP'
                    self.write(fontsz/2, m[3])
                    self.char_vpos = 'LINE'
                case '_':
                    self.char_vpos = 'SUB'
                    self.write(fontsz/2, m[3])
                    self.char_vpos = 'LINE'
        return None

    def addSection(self, sec_title : str, isTitlePage=False):
        self.add_page()
        self.start_section(sec_title)
        self.sections.append(self.Section(sec_title))
        k = len(self.sections)
        if not isTitlePage:
            if self.print_sec_num:
                sec_title = f'{k} ' + sec_title
            self.addParagraph(sec_title, True, False, True, 16, 4)
        self.changed = True

    def addSubsection(self, ss_title : str, section = -1):
        if section >= len(self.sections):
            section = -1
        self.sections[section].append(ss_title)
        k = len(self.sections)
        self.write_html(f'<H2 align="left">{k} {ss_title}</H2>')
        self.changed = True

    def addParagraph(self, text : str = None, bold=False, indent=True, center=False, fontsz=14, extra_lf=0):
        if not isinstance(text, str): return
        self.set_font("FreeSerif", 'B' if bold else '', fontsz)
        if center:
            self.set_x(self.l_margin)
            self.multi_cell(self.epw, txt=text, align='C')
        else:
            self.set_x(self.l_margin + (10 if indent else 0))
            if self.inline_checking:
                self._check_sup_sub(text, fontsz)
            else:
                self.write(fontsz/2, text)
            self.ln(fontsz - 6)
        self.ln(extra_lf)
        self.changed = True

    def addNameValParagraph(self, name, text, fontsz=14, extra_lf=0):
        self.set_font(None, 'B', fontsz)
        if not isinstance(name, str): name = str(name)
        self.write(fontsz/2, name)
        self.set_font(style = '')
        if not isinstance(text, str): text = str(text)
        if not text.startswith(':'): text = ': ' + text
        self.write(fontsz/2, text)
        self.ln(fontsz - 6 + extra_lf)
        self.changed = True

    def addListElem(self, text : str, bold = False, fontsz=14):
        self.set_x(self.l_margin)
        self.set_font(None, 'B' if bold else '', fontsz)
        self.write(fontsz/2, '━ ')
        self.multi_cell(self.epw, fontsz/2, text)
        #self.ln(fontsz-6)
        self.changed = True

    def save(self, filepath : str):
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        self.output(filepath)
        self._log(f'File {filepath} saved')
        self.changed = False
        return self.buffer

    def addCaption(self, caption : str, text: str, fontsz=13):
        if not caption.endswith(' '): caption += ' '
        self.set_font(size=fontsz)
        self.set_x(self.l_margin + self.epw*0.05)
        self.write(fontsz/2, caption)
        if self.inline_checking:
            self._check_sup_sub(text, fontsz)
        else:
            self.multi_cell(self.epw*0.9 - self.x + self.l_margin, fontsz/2, text)
        self.ln(2)
        self.changed = True

    def addImage(self, img : Image.Image, caption : str = None, capfontsz=12, extra_lf=4):
        fig = self.fig_counter + 1
        self.fig_counter = fig
        dt = time.perf_counter()
        if isinstance(img, Figure):
            img.set_tight_layout({"pad": .0})
            canvas = FigureCanvasAgg(img)
            canvas.draw()
            img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        if isinstance(img, Image.Image | str):
            self.set_x(self.l_margin + self.epw*0.05)
            #self.image(img)
            self.image(img, w=self.epw*0.9, keep_aspect_ratio=True)
        if isinstance(caption, str):
            if not caption .endswith('.'): caption += '.'
            self.addCaption(f'Рис.{fig} - ', caption, capfontsz)
        if extra_lf: self.ln(extra_lf)
        dt = time.perf_counter() - dt
        self._log(f'  image {fig} added  [{dt:.3f} sec]')
        self.changed = True

    def addTable(self, _table : list, caption : str = None, col_w = None, fontsz=14, capfontsz=12, extra_lf=4):
        tbl = self.table_counter + 1
        self.addCaption(f'Таблица {tbl} ', caption, capfontsz)
        self.set_font(size=fontsz)
        colws = None if col_w is None else sum(col_w)
        with self.table(width=colws, col_widths=col_w, line_height=1.8*self.font_size) as table:
            for k, trow in enumerate(_table):
                row = table.row()
                row.cell(trow[0], 'L' if k else 'C')
                row.cell(str(trow[1]), 'R' if k else 'C')
        self.ln(extra_lf)
        self.table_counter = tbl
        self.changed = True

    def footer(self):
        if self.disable_first_page_number and self.page == 1: return
        y = self.y
        self.set_xy(self.l_margin, -15)
        self.cell(self.epw, txt=f'{self.page}', align='C')
        self.y = y

    def header(self):
        if self.disable_first_page_number and self.page == 1: return
        y = self.y
        txtclr = self.text_color
        self.set_font(size=12)
        self.set_text_color(180, 180, 180)
        if isinstance(self.organization, str) and len(self.organization) > 0:
            self.set_xy(self.l_margin, 5)
            self.write(txt=self.organization)
        if isinstance(self.title, str) and len(self.title) > 0:
            self.set_xy(self.l_margin + self.epw*0.5, 5)
            self.cell(w=self.epw*0.5, txt=self.title, align='R')
        self.y = y
        self.set_text_color(txtclr)

class ImageTypes(Enum):
    QUATS = 'Компонент кватерниона {c}, измеренного в МЗД (красные точки) и компонент кватерниона, установленного в ОДС (синяя линия)'
    ANG_DIFFS = 'Разница углов ориентации между измерениями в МЗД и установкой в ОДС в приборной системе координат'
    ANG_HIST = 'Гистограммы ошибок углов ориентации Δx, Δy, Δz, представленные в системе координат прибора'
    ANG_DIFFS_AVG = 'Разница углов ориентации между измерениями в МЗД вычисленными средними значениями в приборной системе координат'
    ANG_HIST_AVG = 'Гистограммы отклонений от средних значений углов ориентации Δx,Δy,Δz, представленные в системе координат прибора'
    FREQS = 'Частота выдачи МЗД АЗДК-1 кадров по статистическим данным. Красными точками обозначены «плохие» кадры, синими – прочитанные кадры, зелеными – кадры с успешными отождествлениями'

    @classmethod
    def loadImages(cls, imdir : str, quats=True, hists=True, avgdivs=True, freqs=True, angdiffs=True):
        imgList = []
        if quats:
            for c in PdfFunTestReport.qcomponents:
                #cap = f'Компонент кватерниона {c}, измеренного в МЗД (красные точки) и компонент кватерниона, установленного в ОДС (синяя линия).'
                img = glob(os.path.join(imdir, f'quats*{c}.png'))
                if len(img) > 0: imgList.append((img[0], cls.QUATS))

        allimgs = glob(os.path.join(imdir, 'xyz*.png'))
        img = [x for x in allimgs if 'angles' in x and 'mm' not in x]
        if len(img) > 0 and angdiffs:
            #cap = 'Разница углов ориентации между измерениями в МЗД и установкой в ОДС в приборной системе координат.'
            imgList.append((img[0], cls.ANG_DIFFS))

        img = [x for x in allimgs if 'angles' not in x and 'mm' not in x]
        if len(img) > 0 and hists:
            #cap = 'Гистограммы ошибок углов ориентации Δx,Δy,Δz, представленные в системе координат прибора.'
            imgList.append((img[0], cls.ANG_HIST))

        img = [x for x in allimgs if 'angles' in x and 'mm' in x]
        if len(img) > 0 and angdiffs and avgdivs:
            #cap = 'Разница углов ориентации между измерениями в МЗД вычисленными средними значениями в приборной системе координат.'
            imgList.append((img[0], cls.ANG_DIFFS_AVG))

        img = [x for x in allimgs if 'angles' not in x and 'mm' in x]
        if len(img) > 0 and hists and avgdivs:
            #cap = 'Гистограммы отклонений от средних значений углов ориентации Δx,Δy,Δz, представленные в системе координат прибора.'
            imgList.append((img[0], cls.ANG_HIST_AVG))

        allimgs = glob(os.path.join(imdir, '*frequency*.png'))
        if len(allimgs) > 0 and freqs:
            #cap = 'Частота выдачи МЗД АЗДК-1 кадров по статистическим данным. Красными точками обозначены «плохие» кадры, синими – прочитанные кадры, зелеными – кадры с успешными отождествлениями.'
            imgList.append((allimgs[0], cls.FREQS))
        return imgList

    @classmethod
    def loadStatics1(cls, imdir : str):
        return cls.loadImages(imdir, True, True, True, False, True)

    @classmethod
    def loadStatics2(cls, imdir : str):
        return cls.loadImages(imdir, False, True, True, False, True)

    @classmethod
    def loadStatics3(cls, imdir : str):
        return cls.loadImages(imdir, True, True, False, False, True)

    @classmethod
    def loadDynamics1(cls, imdir : str):
        return cls.loadImages(imdir, True, False, False, True, True)

    @classmethod
    def loadDynamics2(cls, imdir : str):
        return cls.loadImages(imdir, True, False, False, True, False)

class PdfFunTestReport(Pdf):
    qcomponents = 'wxyz'

    def __init__(self, *, verbose=False, author=None, title='АЗДК ФИ', downscale_images=False):
        super().__init__(title, verbose=verbose, downscale_images=downscale_images)
        self.title_page_composed = False
        self.test_counter = 0

    def _add_accuracy_table(self, dtable : DataFrame):
        if dtable is None: return

        tbl = self.table_counter + 1
        self.table_counter = tbl

        self.add_page()
        self.addCaption(f'Таблица {tbl} ', 'Результаты СКО серий измерений с нулевой угловой скоростью')
        self.set_font(size=12)
        col_w = (9, 25, 25, 25, 16, 16, 16, 16, 16, 16)
        with self.table(width=sum(col_w), col_widths=col_w, line_height=1.8*self.font_size) as table:
            row = table.row()
            row.cell()
            row.cell('Углы ориентации, °', 'C', colspan=3)
            row.cell()
            row.cell()
            row.cell('Отклонения, ″', 'C', colspan=3)
            row.cell()
            row.cell()
            row.cell('СКО, ″', 'C', colspan=3)
            row.cell()
            row.cell()
            row = table.row()
            row.cell('№', 'C')
            for c in 'αδφ': row.cell(c, 'C')
            for c in 'xyz': row.cell('Δ' + c, 'C',)
            for c in 'xyz': row.cell('σ' + c, 'C')
            for r, drow in dtable.iterrows():
                row = table.row()
                self.set_font(size=10)
                row.cell(str(r+1), 'R')
                row.cell((f"{drow.ra:.6f}"), 'R')
                row.cell((f"{drow.de:.6f}"), 'R')
                row.cell((f"{drow.phi:.6f}"), 'R')
                row.cell((f"{drow.xa_sh:.1f}"), 'R')
                row.cell((f"{drow.ya_sh:.1f}"), 'R')
                row.cell((f"{drow.za_sh:.1f}"), 'R')
                row.cell((f"{drow.xa_std:.1f}"), 'R')
                row.cell((f"{drow.ya_std:.1f}"), 'R')
                row.cell((f"{drow.za_std:.1f}"), 'R')
            row = table.row()
            row.cell('Средние значения', 'R', colspan=4)
            row.cell()
            row.cell()
            row.cell()
            #
            row.cell(f"{dtable.xa_sh.mean():.1f}", 'R')
            row.cell(f"{dtable.ya_sh.mean():.1f}", 'R')
            row.cell(f"{dtable.za_sh.mean():.1f}", 'R')
            row.cell(f"{np.sqrt((dtable.xa_std**2).mean()):.1f}", 'R')
            row.cell(f"{np.sqrt((dtable.ya_std**2).mean()):.1f}", 'R')
            row.cell(f"{np.sqrt((dtable.za_std**2).mean()):.1f}", 'R')
        self._log(f'  table {tbl} added')

    def _add_statistics_table(self, angvel : float, stats : list, count : int):
        col_w = (int(self.epw*0.45), int(self.epw*0.15))
        n = 0 if stats is None else len(stats)
        _table = [['Статистика', 'Значение'],
                  ['1. Всего кадров измерено', 0 if n < 1 else stats[0]],
                  ['2. Кадров считано', 0 if n < 2 else stats[1]],
                  ['3. Плохих кадров зарегистрировано', 0 if n < 3 else stats[2]],
                  ['4 Кол-во отождествлений', 0 if n < 5 else stats[4]],
                  ['4.1 кол-во нач. отождествлений', 0 if n < 6 else stats[5]],
                  ['4.2 кол-во кадров с ведением', 0 if n < 7 else stats[6]],
                  ['4.3 кол-во срывов ведения', 0 if n < 6 else max(stats[5]//2 - (count or 1), 0)],
                  ['5. Кол-во загрузок из каталога', 0 if n < 9 else stats[8]]]
        self.addTable(_table, f'Статистические результаты функциональных испытаний МЗД АЗДК-1 при скоростях вращения {angvel:.1f}°/сек.', col_w)
        self._log(f'  table {self.table_counter} added')

    def _add_images(self, images : list):
        if images is None or len(images) == 0: return
        # Figure number generator
        figures = iter(range(self.fig_counter + 1, self.fig_counter + 1 + len(images)))
        # quaternion component generator
        qc = iter(self.qcomponents)
        # images cycle
        for _, caption in images:
            fig = next(figures)
            if not isinstance(caption, str):
                if caption == ImageTypes.QUATS:
                    caption = caption.value.format(c=next(qc))
                else:
                    caption = caption.value
            self.addListElem(caption + f' - см. Рис. {fig}.')

        self.add_page()
        qc = iter(self.qcomponents)
        for img, caption in images:
            if not isinstance(caption, str):
                if caption == ImageTypes.QUATS:
                    caption = caption.value.format(c=next(qc))
                else:
                    caption = caption.value
            self.addImage(img, caption)

    @classmethod
    def get_duration(cls, datafile : str):
        print(f'Get durations from {os.path.basename(datafile)}')
        dtable = read_csv(datafile, header=None, sep=';', skiprows=2)
        times = checkTimesOverMidnight([dt.datetime.strptime(x, r'%H:%M:%S.%f') for x in dtable[0]])
        dtimes = [np.round((t2-t1).total_seconds(), -1) for t1, t2 in zip(times, times[1:])]
        dtimes = [x for x in dtimes if x > 0]
        dtimes_set = np.unique(dtimes, return_counts=True)
        k = np.argmax(dtimes_set[1])
        return float(dtimes_set[0][k]), int(dtimes_set[1][k])

    @classmethod
    def get_time_range(cls, datafile : str):
        print(f'Get time range from {os.path.basename(datafile)}')
        colnames=['time', 'cmd', 'val', 'ctime']
        dtable = read_csv(datafile, header=None, sep=';', skiprows=2, names=colnames)
        commands = [' 72h', ' 14h']
        dtable = dtable.loc[dtable.cmd.isin(commands)]
        t_beg = dt.datetime.strptime(dtable.head(1).time._values[0], r'%H:%M:%S.%f')
        t_end = dt.datetime.strptime(dtable.tail(1).time._values[0], r'%H:%M:%S.%f')
        if t_beg > t_end: t_end += dt.timedelta(days=1)
        return t_beg, t_end

    @classmethod
    def _get_dates(cls, datadir : str, azdknum : int = '', pattern='.tracked.'):
        files = glob(os.path.join(datadir, f'*{pattern}*{azdknum}*.txt'))
        min_time, max_time = None, None
        for file in files:
            t = os.path.getmtime(file)
            if min_time is None or t < min_time: min_time = t
            if max_time is None or t > max_time: max_time = t
        return dt.datetime.fromtimestamp(min_time).date(), dt.datetime.fromtimestamp(max_time).date()

    @classmethod
    def get_statistics(cls, datafile : str, time_beg : dt.datetime = None, time_end : dt.datetime = None):
        print(f'Get statistics from {os.path.basename(datafile)}')
        colnames = ['time', 'total', 'read', 'bad', 'processsed', 'identified', 'ii', 'am', 'skipped', 'catalog']
        dtable = read_csv(datafile, header=None, sep=';', skiprows=2, names=colnames)
        times = [dt.datetime.strptime(x, r'%H:%M:%S.%f') for x in dtable.time]
        dtable.time = checkTimesOverMidnight(times)
        k1 = dtable.loc[(dtable.time >= time_beg) * (dtable.time <= time_end) * (dtable.ii.notna())]
        kh = k1.head(1)
        ke = k1.tail(1)
        return [int(k2) - int(k1) for k1, k2 in zip(kh._values[0][1:], ke._values[0][1:])]

    def _duration_text(self, duration : float) -> str:
        if duration < 120:
            duration = f'{duration} секунд'
        elif duration < 7200:
            duration = f'{duration/60:.1f} минуты'
        return duration

    def addTitlePage(self
        , device = 2308, fw_version = '1.7.008A'
        , date_beg = dt.date.today(), date_end = None
        , star_intensity = 3.0, mag_slope = 0.6, star_size = 2.5
        , focus = 33.90, max_mag = 7.5, exp_time = 90
        , bin_mode = True , watchdog = False
        , data_freq = 10, pds_num = 7
        , device_revision = 1.5
        , comm_type = 'RS485'
        , comm_speed = 500
        , cpu_freq = 80
        , place = 'г. Москва, Красная Пресня'
        ):

        if self.title_page_composed: return
        self.title_page_composed = True

        enabled = 'включен'
        disabled = 'выключен'

        if isinstance(date_beg, dt.date):
            date_beg = Date(date_beg)
        if isinstance(date_end, dt.date):
            date_end = Date(date_end)

        self._log('Create title page')

        device = f'{device//100:02d}-{device%100:02d}'

        self.addSection('Титульная страница', True)
        self.addParagraph(f'Отчет о функциональных испытаниях МЗД АЗДК-{device_revision} № {device}', True, False, True, 16, 4)
        self.addParagraph('ООО "Азмерит"', center=True, extra_lf=4)
        self.addNameValParagraph('Объект испытаний', f': малогабаритный звездный датчик АЗДК-{device_revision} (АЗДК-1).')
        self.addNameValParagraph('Метка изделия МЗД АЗДК-1', f': АЗДК-{device_revision} № {device}.')
        self.addNameValParagraph('Цель испытаний', f': выполнить проверку функциональности МЗД АЗДК-1 с версией прошивки {fw_version} с частотой работы процессора {cpu_freq} МГц.')
        self.addNameValParagraph('Место испытаний', f': {place}.')
        _samedate = date_end is None or date_beg == date_end
        self.addNameValParagraph('Время испытаний', f': {date_beg}' if _samedate else f': {date_beg} - {date_end}')
        self.ln(4)
        self.addParagraph('Функциональные испытания МЗД АЗДК 1 проводились с использованием имитатора звездного неба ОДС-1' +
                         f' №{pds_num:04d}, проецировавшего на матрицу МЗД АЗДК 1 динамичные и статичные кадры звездного неба.', extra_lf=6)
        self.addParagraph('Условия:', True, False)
        self.addListElem(f'настройки ПО ОДС: интенсивность {star_intensity:.1f}, влияние величины {mag_slope:.1f}, фокус {focus:.1f} мм, размер изображения звезды {star_size:.1f} пикс;')
        self.addListElem(f'ОДС: каталог звезд до {max_mag:.1f} звездной величины;')
        self.addListElem(f'версия прошивки АЗДК-1: {fw_version};')
        self.addListElem(f'взаимодействие по интерфейсу {comm_type} на скорости {comm_speed} кбит/сек;')
        self.addListElem(f'длительность накопления сигнала в МЗД: {exp_time} мс;')
        self.addListElem('режим бинирования кадров в АЗДК-1 ' + (enabled if bin_mode else disabled) + f' (частота {data_freq} Гц);')
        self.addListElem('сторожевой таймер ' + (enabled if watchdog else disabled) + '.')

    def addAccuracySection(self, count : float = None, duration : float = None,
                           images : list = None, datafile : str = None, angvel : float = None):
        if images is None or len(images) == 0:
            if self.verbose:
                msg = 'Empty image list - skipping accuracy section'
                if count is not None:
                    msg += f' with angvel={angvel}, count={count}, duration={duration}'
                self._log(msg)
            return

        if isinstance(duration, dt.timedelta):
            duration = duration.total_seconds()

        test = self.test_counter + 1
        if self.verbose:
            msg = 'Add accuracy section'
            if count is None: msg += ':'
            else: msg += f' (test-{test}, v={angvel}, count={count}, duration={duration}):'
            self._log(msg)
        self.test_counter = test

        self.addSection(f'Тест-{test} (точностной)')

        duration = self._duration_text(duration)

        if isinstance(datafile, str):
            datafile = read_csv(datafile, header=0, sep=';')

        if bool(angvel):
            frmtxt = f'кадров звездного неба со скоростью {angvel:.1f} °/сек'
        else:
            frmtxt = 'статичных кадров звездного неба'

        self.addParagraph('Задачей теста является измерение точностных характеристик МЗД АЗДК 1.'
                          f' В ходе теста посредством имитатора звездного неба ОДС-1 было подано {count} случайных'
                          f' {frmtxt} длительностью по {duration} каждый.', extra_lf=2)

        self.addParagraph('В результате теста выполнены следующие измерения:')

        if datafile is not None:
            self.addListElem(f'В таблице {self.table_counter + 1} приведены средние значения отклонений кватернионов для всех серий измерений.')

        self._add_images(images)

        self._add_accuracy_table(datafile)

        self._log('  completed')

    def addStatisticsSection(self, count : float = None, duration : float = None,
                             images : list = None, stats : list = None, angvel : float = None):
        
        angvel = round(angvel, 1)

        if images is None or len(images) == 0:
            if self.verbose:
                msg = 'Empty image list - skipping statistics section'
                if count is not None:
                    msg += f' with angvel={angvel}, count={count}, duration={duration}'
                self._log(msg)
            return

        if isinstance(duration, dt.timedelta):
            duration = duration.total_seconds()

        test = self.test_counter + 1
        if self.verbose:
            msg = 'Add statistics section'
            if count is None: msg += ' (orbital):'
            else: msg += f' (test-{test}, v={angvel}, count={count}, duration={duration}):'
            self._log(msg)
        self.test_counter = test

        self.addSection(f'Тест-{self.test_counter} (статистический)')

        if count is None:
            period = duration / (90*60)
            duration = to_hm_str(duration)
            angvel = 4/60
            msg = f'Тест-{test} служит для проверки работы прибора при имитации движения по орбите. '\
                  f'Продолжительность теста составила {duration}, '\
                  f'что соответствует ~{period:.1f} витку аппарата на заданной орбите (МКС).'
        else:
            duration = self._duration_text(duration)
            msg = f'Задачей теста-{test} явлется набор статистики количества отождествленных кадров '\
                   'и количество срывов ведений МЗД АЗДК-1 на случайных участках неба при скорости вращения КА '\
                  f'{angvel} °/сек. Данные получены на основе {count} случайных звездных треков длительностью по {duration}.'

        self.addParagraph(msg, extra_lf=2)

        if stats is not None:
            self._add_statistics_table(angvel, stats, count)

        self.addParagraph('В результате теста выполнены следующие измерения:')
        self._add_images(images)

        self._log('  completed')

class FunTestPdfCreator(AzdkThread):
    class Task(Enum):
        add_title = 0
        add_accuracy = 1
        add_statistics = 2
        finish = 3

    def __init__(self, autosave=300.0, verbose=False, logger=None, author : str = None, title='АЗДК ФИ'):
        super().__init__(self.__class__.__name__, verbose, logger)
        self.fonts = {}
        self.sectionTasts = []
        self.autosave = autosave
        self.autosave_path = os.environ['TEMP'] + '/pdfcreator_temp.pdf'
        self.author = author
        self.title = title
        self.currentTask = None
        self.downscale_images = True
        self.output = b''
        self.installFont('FreeSerif',
            'D:/Users/Simae/PyWork/data/fonts/FreeSerif.otf',
            'D:/Users/Simae/PyWork/data/fonts/FreeSerifBold.otf')

    def installFont(self, famaly : str, regularPath : str, boldPath : str):
        self._mutex.acquire()
        self.fonts[famaly]=(regularPath, boldPath)
        self._mutex.release()

    def addTask(self, st : Task, **kwargs):
        self.log(f'Enqueue {st.name} section')
        self._mutex.acquire()
        self.sectionTasts.append((st, kwargs))
        self._mutex.release()

    def enqueueTitleSection(self, **kwargs):
        self.addTask(self.Task.add_title, **kwargs)

    def enqueueAccuracySection(self, **kwargs):
        self.addTask(self.Task.add_accuracy, **kwargs)

    def enqueueStatisticsSection(self, **kwargs):
        self.addTask(self.Task.add_statistics, **kwargs)

    def enqueueFinish(self):
        self.addTask(self.Task.finish)

    def run(self):
        self.running = True
        pdf = PdfFunTestReport(verbose=self.verbose, author=self.author, title=self.title,
                               downscale_images=self.downscale_images)
        pdf.print_sec_num = False
        pdf.organization = 'ООО Азмерит'

        autoSaveTimer = time.perf_counter()

        while self.running:
            time.sleep(0.01)
            self._mutex.acquire()
            for family, dirs in self.fonts.items():
                try:
                    pdf.add_font(family, '', dirs[0])
                    pdf.add_font(family, 'B', dirs[1])
                    pdf.set_font(family)
                except FileNotFoundError: pass
            self.fonts.clear()
            try: self.currentTask, params = self.sectionTasts.pop(0)
            except IndexError: self.currentTask = None
            self._mutex.release()
            match self.currentTask:
                case self.Task.add_title:
                    pdf.addTitlePage(**params)
                case self.Task.add_accuracy:
                    pdf.addAccuracySection(**params)
                case self.Task.add_statistics:
                    pdf.addStatisticsSection(**params)
                case self.Task.finish:
                    if pdf.changed and self.autosave_path:
                        self.output = pdf.save(self.autosave_path)
                    pdf = PdfFunTestReport(verbose=self.verbose, author=self.author)
            self.currentTask = None
            #if isinstance(self.autosave, float):
               # t = time.perf_counter()
               # if t > autoSaveTimer + self.autosave:
                #    autoSaveTimer = t
                #    if pdf.changed and self.autosave_path:
                #        self.output = pdf.save(self.autosave_path)

        if pdf.changed and self.autosave_path:
            self.output = pdf.save(self.autosave_path)

        self.running = False

    def sync(self, stop=False):
        while not self.is_alive() or not self.running:
            time.sleep(0.1)
        while len(self.sectionTasts) > 0 or self.currentTask is not None:
            time.sleep(0.1)
        if stop: self.stop()

def fpdftest3(rdir, azdknum=2101, dirsfx='', freq=5, exptime=100, fw_ver='1.07.008A', rs485=500):
    datadir = rdir + f'{azdknum}{dirsfx}/'
    logfile = datadir + 'pdfcreator.log'

    pdfc = FunTestPdfCreator(verbose=True, logger=logfile, author='Тучин М. С.')
    pdfc.autosave_path = rdir + f'ФИ АЗДК-1.5 №{azdknum//100:02d}-{azdknum%100:02d}{dirsfx}.pdf'
    if not pdfc.waitUntilStart(): return

    dates = PdfFunTestReport._get_dates(datadir, azdknum)
    pdfc.enqueueTitleSection(device=azdknum, date_beg=dates[0], date_end=dates[1], bin_mode=freq>4,
                         data_freq=freq, exp_time=exptime, fw_version=fw_ver, comm_speed=rs485)
    # add static section
    imgs = ImageTypes.loadStatics1(datadir + f'azdk{azdknum}s')
    if len(imgs) > 0:
        duration, count = PdfFunTestReport.get_duration(datadir + f'pdsserver.tracked.azdk{azdknum}s.txt')
        pdfc.enqueueAccuracySection(count=count, duration=duration, images=imgs, datafile=datadir + f'azdk{azdknum}s/errors.csv')
    # add static section
    imgs = ImageTypes.loadStatics2(datadir + f'azdk{azdknum}ss')
    if len(imgs) > 0:
        duration, count = PdfFunTestReport.get_duration(datadir + f'pdsserver.tracked.azdk{azdknum}ss.txt')
        pdfc.enqueueAccuracySection(count=count, duration=duration, images=imgs)
    # add static section
    imgs = ImageTypes.loadImages(datadir + f'azdk{azdknum}r01', avgdivs=False)
    if len(imgs) > 0:
        duration, count = PdfFunTestReport.get_duration(datadir + f'pdsserver.tracked.azdk{azdknum}r01.txt')
        _times = PdfFunTestReport.get_time_range(datadir + f'azdkserver.tracked.azdk{azdknum}r01.txt')
        _stats = PdfFunTestReport.get_statistics(datadir + f'azdkclient.tracked.azdk{azdknum}r01.txt', *_times)
        pdfc.enqueueStatisticsSection(count=count, duration=duration, images=imgs, stats=_stats, angvel=0.1)
    # add dynamic sections
    for angvel in range(1, 4):
        imgs = ImageTypes.loadDynamics1(datadir + f'azdk{azdknum}r{angvel}')
        if len(imgs) == 0: continue
        duration, count = PdfFunTestReport.get_duration(datadir + f'pdsserver.tracked.azdk{azdknum}r{angvel}.txt')
        _times = PdfFunTestReport.get_time_range(datadir + f'azdkserver.tracked.azdk{azdknum}r{angvel}.txt')
        _stats = PdfFunTestReport.get_statistics(datadir + f'azdkclient.tracked.azdk{azdknum}r{angvel}.txt', *_times)
        pdfc.enqueueStatisticsSection(count=count, duration=duration, images=imgs, stats=_stats, angvel=angvel)
    # orbital
    imgs = ImageTypes.loadDynamics1(datadir + f'azdk{azdknum}o')
    if len(imgs) > 0:
        _times = PdfFunTestReport.get_time_range(datadir + f'azdkserver.tracked.azdk{azdknum}o.txt')
        _stats = PdfFunTestReport.get_statistics(datadir + f'azdkclient.tracked.azdk{azdknum}o.txt', *_times)
        pdfc.enqueueStatisticsSection(duration=_times[1]-_times[0], images=imgs, stats=_stats)
    # finish (save)
    pdfc.enqueueFinish()

    print('*** Wait for execution ***')
    pdfc.sync(True)

def fpdftest4():
    wdir = 'D:/Users/Simae/PyWork/data/'

    pdf = Pdf()
    pdf.inline_checking = True
    ffam = 'FreeSerif'
    fdir = wdir + 'freefont-20120503/FreeSerif'
    pdf.add_font(ffam, '', fdir + '.otf')
    pdf.add_font(ffam, 'B', fdir + 'Bold.otf')
    pdf.set_font(ffam)

    pdf.addSection('Section One')
    txt = 'Гистограммы отклонений от средних значений углов ориентации Δ_x^2, Δ_y^2, Δ_z^2, представленные в системе координат прибора'
    pdf.addParagraph(txt)
    pdf.save(wdir + 'pdf_test.pdf')

if __name__ == "__main__":
    rdir = 'd:/Users/Simae/Work/2023/tochmash_func_tests/'
    fpdftest3(rdir, 2312, 'a', 10, 90, '1.07.00A8', 921.6)
    #fpdftest4()

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None or gettrace() is None:
        os.system('pause')
