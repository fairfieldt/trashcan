import json,pathlib,time
smc=pathlib.Path('/sys/devices/platform/applesmc.768')
for i in range(30):
 row={'unix':time.time(),'fans':{k:int((smc/k).read_text()) for k in ['fan1_min','fan1_max','fan1_manual','fan1_output','fan1_input']}}
 row['gpu_temperatures']={str(p):int(p.read_text())/1000 for p in pathlib.Path('/sys/class/drm').glob('card[0-9]/device/hwmon/hwmon*/temp1_input')}
 print(json.dumps(row),flush=True)
 time.sleep(1)
