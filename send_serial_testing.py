import serial
arduino = serial.Serial('COM5', baudrate=9600, timeout=1)
def flash_light(label):
    arduino.write(1)

