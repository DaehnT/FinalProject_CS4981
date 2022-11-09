#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
from pybricks.messaging import BluetoothMailboxServer, TextMailbox

from _thread import allocate_lock

import time

DEG_PER_SEC = 90

# Assume arm starts down, claw is open and directly in front of color sensor
ev3 = EV3Brick()
claw_motor = Motor(Port.A)
raise_arm_motor = Motor(Port.B)
spin_arm_motor = Motor(Port.C)

raise_arm_motor.reset_angle(0)
spin_arm_motor.reset_angle(0)
claw_motor.reset_angle(0)

# Write your program here.
ev3.speaker.set_volume(400, which='_all_')
ev3.speaker.beep()

#Note the timeout parameter doesn't seem to work
#This wait checks for a message and then just returns
#Compared to wait() which will sit there and wait
#for a message
def wait_with_timeout(mbox,name):
        """Waits until ``mbox`` receives a value."""
        lock = allocate_lock()
        lock.acquire()
        with mbox._connection._lock:
            mbox._connection._updates[name] = lock
        try:
            #timeout parameter doesn't work as
            #expected so I'm just setting
            #the wait flag to be zero
            return lock.acquire(0,1)
        finally:
            with mbox._connection._lock:
                del mbox._connection._updates[name]

def ping():
    server = BluetoothMailboxServer()
    mbox = TextMailbox('greeting', server)

    # The server must be started before the client!
    print('waiting for connection...')
    server.wait_for_connection()
    print('connected!')

    # In this program, the server waits for the client to send the first message
    # and then sends a reply.
    mbox.wait()
    print(mbox.read())
    mbox.send('hello to you!')

def setupConnection():
    server = BluetoothMailboxServer()
    mbox = TextMailbox('greeting', server)

    # The server must be started before the client!
    print('waiting for connection...')
    server.wait_for_connection()
    print('connected!')
    return mbox

def receiveMsg():
    mbox = setupConnection()
    while 1:
        mbox = setupConnection()
        mbox.wait()
        msg = mbox.read()
        print("Message received is: "+msg)
        if msg == "q":
            break

def server():
    mbox = setupConnection()
    pressed = ""
    while 1:
        wait_with_timeout(mbox,mbox.name) #non-blocking
        msg = mbox.read()
        #if pressed != []:
        #    print("pressed  is "+str(pressed))
        if msg != None:
            print("Message received is: "+msg)
            #assumes there are only two tokens
            cmd,arg = mbox.read().split(":")
            if cmd == "raise":
                raise_arm_motor.run_target(DEG_PER_SEC, -270)
                mbox.send('ready')
            elif cmd == "lower":
                raise_arm_motor.run_target(DEG_PER_SEC, 0)
                mbox.send('ready')
            elif cmd == "close":
                claw_motor.run_target(DEG_PER_SEC, 90)
                mbox.send('ready')
            elif cmd == "open":
                claw_motor.run_target(DEG_PER_SEC, 0)
                mbox.send('ready')
            elif cmd == "grab":
                claw_motor.run_until_stalled(DEG_PER_SEC)
                claw_motor.run_time(DEG_PER_SEC, .3)  # Grips object tighter
                mbox.send('ready')
            elif cmd == "rotate":
                # 0 degrees is directly infront of color sensor
                # gear ratio is 12/36
                angle = float(arg)
                spin_arm_motor.run_target(-DEG_PER_SEC, -angle*3)
                mbox.send('ready')
            else:
                print("Received invalid cmd "+cmd)
            #clears msg and the mailbox
            msg = None
            mbox._connection._mailboxes={}
        
        buttons = ev3.buttons.pressed()
        #gets the buttons pressed if any
        if buttons != []:
            pressed = "" #clears pressed
            for i in range(len(buttons)):
                pressed+=str(buttons[i])
                if i!=len(buttons)-1:
                    pressed+=":"
                    
def main():
    server()

if __name__ == '__main__':
    main()