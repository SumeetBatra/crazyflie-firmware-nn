# -*- coding: utf-8 -*-
"""
example on how to plot decoded sensor data from crazyflie
@author: jsschell
"""
import CF_functions as cff
import matplotlib.pyplot as plt
import re
import csv
import numpy as np
# decode binary log data
logData = cff.decode("/media/artem/06BF-215C/log00")

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'
# --- Uncomment To save data as csv File -----------

# to_csv = np.array([[logData['tick']],[logData['stateEstimate.x']],
# [logData['stateEstimate.y']],[
# logData['stateEstimate.z']],[logData['stateEstimate.vx']],
# [logData['stateEstimate.vy']],
# [logData['stateEstimate.vz']],[logData['stateEstimate.qx']],
# [logData['stateEstimate.qy']],
# [logData['stateEstimate.qz']],
# [logData['stateEstimate.qw']],[logData['gyro.x']],
# [logData['gyro.y']],
# [logData['gyro.z']]])

# with open('/home/artem/Sunny/Results/previous_step_50Hz_5_steps.csv','w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(zip(*to_csv[0],*to_csv[1],*to_csv[2],*to_csv[3],*to_csv[4],*to_csv[5],*to_csv[6],*to_csv[7],*to_csv[8],*to_csv[9],*to_csv[10],*to_csv[11],*to_csv[12],*to_csv[13]))
# csvFile.close()
    
# number of columns and rows for suplot
plotCols = 1;
plotRows = 1;

# let's see which keys exists in current data set
keys = ""
for k, v in logData.items():
    keys += k

# get plot config from user
plotGyro = 0
if re.search('gyro', keys):
    inStr = input("plot gyro data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotGyro = 1
        plotRows += 1

plotAccel = 0
if re.search('acc', keys):
    inStr = input("plot accel data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotAccel = 1
        plotRows += 1

plotMag = 0
if re.search('mag', keys):
    inStr = input("plot magnetometer data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotMag = 1
        plotRows += 1

plotBaro = 0
if re.search('baro', keys):
    inStr = input("plot barometer data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotBaro = 1
        plotRows += 1

plotCtrl = 0
if re.search('ctrltarget', keys):
    inStr = input("plot control data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotCtrl = 1
        plotRows += 1

plotStab = 0
if re.search('stabilizer', keys):
    inStr = input("plot stabilizer data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotStab = 1
        plotRows += 1

plotStateEstimate = 0
if re.search('stateEstimate', keys):
    inStr = input("plot stateEstimate data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotStateEstimate = 1
        plotRows += 3

plotIn = 0
if re.search('in', keys):
    inStr = input("plot in data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotIn = 1
        # plotRows += 3
    
# current plot for simple subplot usage
plotCurrent = 0;

# new figure
plt.figure(0)

if plotGyro:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['gyro.x'], '-', label='X')
    plt.plot(logData['tick'], logData['gyro.y'], '-', label='Y')
    plt.plot(logData['tick'], logData['gyro.z'], '-', label='Z')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Gyroscope [°/s]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)
 
if plotAccel:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['acc.x'], '-', label='X')
    plt.plot(logData['tick'], logData['acc.y'], '-', label='Y')
    plt.plot(logData['tick'], logData['acc.z'], '-', label='Z')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Accelerometer [g]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)
 

if plotMag:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['mag.x'], '-', label='X')
    plt.plot(logData['tick'], logData['mag.y'], '-', label='Y')
    plt.plot(logData['tick'], logData['mag.z'], '-', label='Z')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Magnetometer')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

if plotBaro:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['baro.pressure'], '-')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Pressure [hPa]')
    
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['baro.temp'], '-')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Temperature [degC]')

if plotCtrl:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['ctrltarget.roll'], '-', label='roll')
    plt.plot(logData['tick'], logData['ctrltarget.pitch'], '-', label='pitch')
    plt.plot(logData['tick'], logData['ctrltarget.yaw'], '-', label='yaw')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Control')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

if plotStab:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['stabilizer.roll'], '-', label='roll')
    plt.plot(logData['tick'], logData['stabilizer.pitch'], '-', label='pitch')
    plt.plot(logData['tick'], logData['stabilizer.yaw'], '-', label='yaw')
    plt.plot(logData['tick'], logData['stabilizer.thrust'], '-', label='thrust')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('Stabilizer')
    plt.legend(loc=9, ncol=4, borderaxespad=0.)

if plotStateEstimate:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)

    plt.plot(logData['tick'], logData['stateEstimate.x'], '-', label='X')
    plt.plot(logData['tick'], logData['stateEstimate.y'], '-', label='Y')
    plt.plot(logData['tick'], logData['stateEstimate.z'], '-', label='Z')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('stateEstimate Position')
    plt.legend(loc=9, ncol=4, borderaxespad=0.)

if plotStateEstimate:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['stateEstimate.vx'], '-', label='VX')
    plt.plot(logData['tick'], logData['stateEstimate.vy'], '-', label='VY')
    plt.plot(logData['tick'], logData['stateEstimate.vz'], '-', label='VZ')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('stateEstimate Velocity')
    plt.legend(loc=9, ncol=4, borderaxespad=0.)

if plotStateEstimate:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['tick'], logData['stateEstimate.qx'], '-', label='qx')
    plt.plot(logData['tick'], logData['stateEstimate.qy'], '-', label='qy')
    plt.plot(logData['tick'], logData['stateEstimate.qz'], '-', label='qz')
    plt.plot(logData['tick'], logData['stateEstimate.qw'], '-', label='qw')
    plt.xlabel('RTOS Ticks')
    plt.ylabel('stateEstimate Quat')
    plt.legend(loc=9, ncol=4, borderaxespad=0.)

plt.show()