import json,pathlib,subprocess,time
smc=pathlib.Path('/sys/devices/platform/applesmc.768')
keys=['fan1_min','fan1_max','fan1_manual','fan1_output','fan1_input']
def read():return {k:int((smc/k).read_text().strip()) for k in keys}
print(json.dumps({'phase':'before','unix':time.time(),'fans':read()}),flush=True)
try:
 subprocess.run(['systemctl','stop','fancurve.service'],check=True)
 state=read(); target=state['fan1_max']; assert state['fan1_min'] <= target <= 16383,state
 (smc/'fan1_manual').write_text('1\n'); (smc/'fan1_output').write_text(str(target)+'\n')
 print(json.dumps({'phase':'set','unix':time.time(),'requested_rpm':target,'fans':read()}),flush=True)
 for i in range(30):
  time.sleep(1)
  print(json.dumps({'phase':'measure','unix':time.time(),'fans':read()}),flush=True)
finally:
 # Restore the known service after this bounded probe; deployment is a separate step.
 subprocess.run(['systemctl','start','fancurve.service'],check=True)
 print(json.dumps({'phase':'restored_service','unix':time.time(),'fans':read()}),flush=True)
