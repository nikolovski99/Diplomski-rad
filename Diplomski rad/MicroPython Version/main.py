from machine import Pin
from utime import sleep
import rp2
import network
import sys
import config
# Import the desired application. Note: all the applicatins are started by: app.run_main().
import mp_async_example7_NN_MNIST as app

def connect( ssid, pswd ):
    rp2.country('RS')   # Set country Republic Serbia
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.config(pm = 0xa11140) # Disable power-save mode
    wlan.connect(ssid, pswd)
    print("conecting to wifi...")
    while not wlan.isconnected():
        pass
    ifc = wlan.ifconfig()
    print( ifc )
    config.cfg["myIP"] = ifc[0]
    print('config.cfg["myIP"] =', config.cfg["myIP"])
    return wlan
"""
def connect1( ssid, pswd ):
    # This function should be more robust, but in our experiments it made no difference to the function connect above.
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, pswd)
    print("conecting to wifi...")

    # Wait for connect or fail
    max_wait = 10
    while max_wait > 0:
        if wlan.status() < 0 or wlan.status() >= 3:
            break
        max_wait -= 1
        print('waiting for connection...')
        sleep(1)
    
    # Handle connection error
    if wlan.status() != 3:
        raise RuntimeError('network connection failed')
    
    ifc = wlan.ifconfig()
    print( ifc )
    config.cfg["myIP"] = ifc[0]
    print('config.cfg["myIP"] =', config.cfg["myIP"])
    return wlan
"""
def ledblink(nblinks):
    pin = Pin("LED", Pin.OUT)
    print("LED starts flashing...")
    for b in range(nblinks):
        pin.toggle()
        sleep(1) # sleep 1sec

def setargv(args):
    for arg in args:
        sys.argv.append(arg)

pin = Pin("LED", Pin.OUT)
pin.value(0)
ledblink(5)
pin.value(0)

wlan = connect(
    config.cfg["wlan"]["ssid"],
    config.cfg["wlan"]["pswd"]
)

setargv(config.cfg["argv"])
print('sys.argv=', sys.argv)

pin.value(1)
app.run_main()

print("The end. Turn off the LED...")
pin.value(0)
while True:
    sleep(1) # sleep 1sec while waiting for reset or power off
