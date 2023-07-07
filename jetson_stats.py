from jtop import jtop
import datetime
import time

with jtop() as jetson:
    while jetson.ok():
        record = {}

        record['time'] = jetson.stats['time'].timestamp()
        record['GPU'] = jetson.stats['GPU']
        record['CPU1'] = jetson.stats['CPU1']
        record['CPU2'] = jetson.stats['CPU2']
        record['CPU3'] = jetson.stats['CPU3']
        record['CPU4'] = jetson.stats['CPU4']
        record['CPU5'] = jetson.stats['CPU5']
        record['CPU6'] = jetson.stats['CPU6']
        record['Temp CPU'] = jetson.stats['Temp CPU']
        record['Temp GPU'] = jetson.stats['Temp GPU']

        for name, mem_data in jetson.memory.items():
            if name == "RAM":
                record['RAM_Total'] = mem_data['tot']
                record['RAM_Usage'] = mem_data['used']

        
        print(record)
