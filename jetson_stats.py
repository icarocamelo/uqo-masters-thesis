from jtop import jtop
import datetime
import time

with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
        # Print all cpu
        for name, data in jetson.memory.items():
            # print(jetson.stats)

            if name == "RAM":
                # print("------ {name} ------".format(name=name))
                ct = datetime.datetime.now()
                # print("current time:-", ct)
                data['ts'] = ct.timestamp()
                print(data)
                time.sleep(5)

#with jtop() as jetson:
#    # jetson.ok() will provide the proper update frequency
#    while jetson.ok():
#        # Read tegra stats
#        print(jetson.stats)
