#!/usr/bin/env python3
from pybricks.messaging import BluetoothMailboxClient, TextMailbox
import controller as brain
import constants as env

import time

DEG_PER_SEC = 100

SERVER = '24:71:89:4A:5C:59'
def ping():
    client = BluetoothMailboxClient()
    mbox = TextMailbox('greeting', client)

    print('establishing connection...')
    client.connect(SERVER)
    print('connected!')
    mbox.send('hello!')
    mbox.wait()
    print(mbox.read())

def sendMsg():
    client = BluetoothMailboxClient()
    mbox = TextMailbox('greeting', client)

    print('establishing connection...')
    client.connect(SERVER)
    print('connected!')
    while 1:
        cmd = input("Enter a letter: ")
        print("Sent command was: "+cmd)
        mbox.send(cmd)

def client():
    client = BluetoothMailboxClient()
    mbox = TextMailbox('greeting', client)
    robot.se
    print('establishing connection...')
    client.connect(SERVER)
    print('connected!')
    while 1:
        cmd = input("Enter a cmd (raise, lower, close, open, grab, rotate, color): ")
        if cmd == 'raise':
            mbox.send(cmd + ':')
        elif cmd == 'lower':
            mbox.send(cmd + ':')
        elif cmd == 'close':
            mbox.send(cmd + ':')
        elif cmd == 'open':
            mbox.send(cmd + ':')
        elif cmd == 'grab':
            mbox.send(cmd + ':')
        elif cmd == 'rotate':
            degrees = input("Rotate to what degree angle: ")
            msg = cmd + ':' + degrees
            mbox.send(msg)
        elif cmd == 'color':
            mbox.send(cmd + ':')
            mbox.wait()
            color = mbox.read()
            print("Color detected: ", color)
        else:
            print("Unrecognized cmd: "+cmd)

def sort(num_objects, angles, sort_color):
    client = BluetoothMailboxClient()
    mbox = TextMailbox('greeting', client)

    print('establishing connection...')
    client.connect(SERVER)
    print('connected!')
    mbox.send('raise:')
    mbox.wait()
    for i in range(num_objects):
        mbox.send('rotate:' + str(angles[i]))
        mbox.wait()
        mbox.send('lower:')
        mbox.wait()
        mbox.send('grab:')
        mbox.wait()
        mbox.send('raise:')
        mbox.wait()
        mbox.send('rotate:0')
        mbox.wait()
        mbox.send('lower:')
        mbox.wait()
        mbox.send('color:')
        mbox.wait()
        color = mbox.read()
        print('Color detected: ', color)
        mbox.send('raise:')
        mbox.wait()
        if color == 'Color.' + sort_color:
            mbox.send('rotate:190')
        else:
            mbox.send('rotate:-30')
        mbox.wait()
        mbox.send('open:')
        mbox.wait()
    mbox.send('rotate:0')
    mbox.wait()
    mbox.send('lower:')
    mbox.wait()

def main():
    sort(3, (40, 79, 121), 'GREEN')

if __name__ == '__main__':
    main()
