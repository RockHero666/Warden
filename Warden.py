from azdk.azdksocket import AzdkSocket, PDSServerCommands, AzdkServerCommands, AzdkServerCmd, PDSServerCmd , call_azdk_cmd
from threading import Thread
import os
import time
from azdk.azdkdb import AzdkDB
from AzdkCommands import AzdkCommands
import platform
import re
import psutil
from azdk.utils import AzdkLogger
from tqdm import tqdm
from presset.azdk_fun_test import azdk_fun_tests
from datetime import datetime as dt
from azdk.linalg import Quaternion, Vector
import numpy as np


class presset(Thread):
     
     def __init__(self, wdir = None, azs : AzdkSocket = None, ods : AzdkSocket = None ):
         super().__init__()
         self.azs = azs
         self.ods = ods
         self.wdir = wdir
         self.host_path = None

     def set_servers(self, azs : AzdkSocket = None, ods : AzdkSocket = None):
         self.azs = azs
         self.ods = ods

     def set_wdir(self, wdir = None):
        self.wdir = wdir
     
     def set_azdkHOST_path(self,host_path = None):
        self.host_path = host_path

        
     def start_test(self):
        azdk_fun_tests(duration = 6,count=6,wdir=self.wdir,azdkdbpath=self.host_path,ip="25.21.118.38")
                



class warden():

    def __init__(self, press : presset):

        current_file = os.path.abspath(__file__)
        self.directory = os.path.dirname(current_file)
        self.parent_directory = (os.path.dirname(self.directory))

        self.name = "Warden"
        self.azs = AzdkSocket()
        self.ods = AzdkSocket()
        logpath = self.directory + '/logs/Warden_' + dt.now().strftime(r'%Y.%m.%d_%H.%M.%S') + '.log'
        self.logger = AzdkLogger(logpath)
        
        self.press = press
        self.press.set_servers(self.azs,self.ods)
        self.press.set_wdir(f"{self.directory}/presset/wdir/")
        self.press.set_azdkHOST_path(f"{self.directory}/presset/AZDKHost.xml")
        
        self.connect_AZDK = ["25.21.118.38",56001]
        self.connect_ODS = ["25.21.118.38",55555]
        self.processes = {"azdkserver.exe" : None, "PDSServer_nogui.exe" : None}
        self.db = AzdkDB('../AZDKHost.xml')
        self.is_runing = True
        self.global_timeout = 5
        self.system = platform.system()
        self.only_once = True

        self.azdk_server_check_cmd = AzdkServerCmd(AzdkServerCommands.GET_VERSION, None, None,timeout=self.global_timeout)
        self.ods_server_check_cmd = PDSServerCmd(PDSServerCommands.GET_STATE, None, None,timeout=self.global_timeout)
        self.azdk_state_cmd = self.db.createcmd(AzdkCommands.READ_FW_VERSION.value, None, timeout=self.global_timeout)
        self.ods_state_cmd = PDSServerCmd(PDSServerCommands.GET_DISPLAY, None, None,timeout=self.global_timeout) 

    def end_track(self):
        initQuat = Quaternion.random()
        initAngVel = Vector.randomdir() * (3.0*np.pi/180)

        devAngVel = initQuat.rotateInverse(initAngVel).data()*(2**30)
        devAngVel = devAngVel.astype(np.int32)

        cmd = self.db.createcmd(7, devAngVel, timeout=self.global_timeout*0.5)
        call_azdk_cmd(self.azs, cmd, self.global_timeout)

        self.ods.execute(PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(0), timeout=self.global_timeout*0.5), self.global_timeout)

        pdsSetQuat = PDSServerCmd(PDSServerCommands.SET_ORIENT, initQuat, timeout=self.global_timeout*0.5)
        self.ods.execute(pdsSetQuat, self.global_timeout)

        pdsSetAngVel = PDSServerCmd(PDSServerCommands.SET_ANGVEL, initAngVel, timeout=self.global_timeout*0.5)
        self.ods.execute(pdsSetAngVel, self.global_timeout)

        fl = PDSServerCommands.STATE_ROTATION_ON.value | PDSServerCommands.STATE_SHOW_ON.value
        cmd = PDSServerCmd(PDSServerCommands.SET_STATE, np.uint32(fl), timeout=self.global_timeout*0.5)
        while not self.ods.execute(cmd, self.global_timeout*2):
            pass
             
        self.logger.log(f"Установлен конечный трек для ожидания {initQuat}  {initAngVel} Angvel = 3.0")

    def find_process_by_name(self, name):
        for process in psutil.process_iter(['name']):
            if process.name().lower() == name.lower():

                return process
        return None

    def servers_start_windows(self): #включаем серваки
        process_names = ['azdkserver.exe', 'PDSServer_nogui.exe']
        
        self.logger.log("Запуск серверов")

        for process_name in process_names:
            process_path = self.parent_directory + "/" + process_name   
            if self.find_process_by_name(process_name) is None: 


                os.chdir(self.parent_directory)
                os.startfile(process_path)
                self.logger.log(f"Сервер {process_name} запущен")
                os.chdir(self.directory) 

    def servers_start_linux(self):
        pass

    def connect_server(self): # подключаем серваки
        self.azs = AzdkSocket(self.connect_AZDK[0], self.connect_AZDK[1], AzdkServerCommands, threadName="Warden_azdk", verbose= False)
        if not self.azs.waitUntilStart():
            self.logger.log(f"Ошибка подключения AzdkServer")
            self.azs = None
            return False
        else:
            self.logger.log(f"Подключение к AzdkServer установленно")
            self.azs.setConnectionName("Warden",self.global_timeout)

        self.ods = AzdkSocket(self.connect_ODS[0], self.connect_ODS[1], PDSServerCommands, threadName="Warden_ods", verbose= False)
        if not self.ods.waitUntilStart():
            self.logger.log(f"Ошибка подключения ODSServer")
            self.ods = None
            return False
        else:
            self.logger.log(f"Подключение к ODSServer установленно")
            self.ods.setConnectionName("Warden",self.global_timeout)

        return True
        
    def servers_state(self): # Проверяем серваки
        self.azs.enqueue(self.azdk_server_check_cmd)
        azs_answer = self.azs.waitforanswer(self.azdk_server_check_cmd, self.global_timeout*1.5)
        
        try:
           if re.match(r"\d\.\d+\.[0-9a-fA-F]{4,4}$", azs_answer.answer[0]) is not None:
               pass
           else:
               self.logger.log(f"AzdkServer не дал ответа или ответ неверен {azs_answer.answer[0]}")
               return False
        except (IndexError , TypeError, AttributeError) as error:
            self.logger.log(f"Ответ AzdkServer вызвал исключение {error}")
            return False

        self.ods.enqueue(self.ods_server_check_cmd)
        ods_answer = self.ods.waitforanswer(self.ods_server_check_cmd, self.global_timeout*1.5)
        
        try:
          return isinstance(ods_answer.answer[0], int)
        except (AttributeError,IndexError) as error:
            self.logger.log(f"Ответ ODSServer вызвал исключение {error}")
            return False

    def azdk_ods_state(self):
        
        time.sleep(0.2)
        call_azdk_cmd(self.azs, self.azdk_state_cmd, timeout= self.global_timeout*1.5)
        answ = self.db.answer(self.azdk_state_cmd) 
        try:
            if isinstance(answ, bytes) and len(answ) == 4:
                pass
            else:
                self.logger.log(f"Azdk не дал ответа или ответ неверен {answ}")
                return False
        except (IndexError , TypeError, AttributeError) as error :
            self.logger.log(f"Ответ Azdk вызвал исключение {error}")
            return False
        self.ods.enqueue(self.ods_state_cmd)
        ods_answer = self.ods.waitforanswer(self.ods_state_cmd, self.global_timeout*1.5)
        try:
            if ods_answer.answer[1] == "OLT0001 [1280x1024]":
                pass
            else:
                self.logger.log(f"Ответ ODS неверен {ods_answer.answer[1]}")
                return False
        except (IndexError , TypeError, AttributeError) as error :
            self.logger.log(f"Ответ ODS вызвал исключение {error}")
            return False
        return True
        
    def monitor(self) -> None:
        self.logger.log("Начало работы Warden`a")
        runing_level = 1
        test_gate = True
        while self.is_runing:
            self.logger.flush()
            match runing_level:
                case 1:
                    if self.system == "Windows":
                        self.servers_start_windows()
                        if self.connect_server():
                            runing_level+=1
                            self.logger.log(f"Level up")
                        else:
                            test_gate = True
                            
                    elif self.system == "Linux":
                        self.servers_start_linux()
                        if self.connect_server():
                            runing_level+=1
                        else:
                            test_gate = True
                    else:
                        self.logger.log(f"Level down, bad OS")
                        print("Не удалось определить операционную систему.")
                case 2:
                    if self.servers_state():
                        runing_level+=1
                        self.logger.log(f"Level up, сервера дали ответ")
                    else:
                        runing_level-=1
                        self.logger.log(f"Level down, серверы не ответили")
                        test_gate = True
                case 3:
                    if self.azdk_ods_state():
                        if self.only_once :
                            self.logger.log(f"Начало испытаний")
                            self.press.start_test()
                            self.only_once = False
                            self.end_track()
                    else:
                        self.logger.log(f"Level down, оборудование не дало ответ")
                        runing_level-=1
                        test_gate = True


    
    def test(self):
        for i in tqdm(range(100), ncols=80, ascii=True, desc='Total'):
            time.sleep(0.1)


        



if __name__ == '__main__':
    press = presset()
    ward = warden(press)
    #azdk_fun_test_results_analyzer(device=2301, wdir='D:/AZDK', wdir_sfx='', freq=5,
                                   #version='1.05.00A5', pds_num=2304)
    
    ward.monitor()
   
    #start_presset()
