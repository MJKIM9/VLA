import struct
import serial


def _crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return struct.pack('<H', crc)


def _fc06_frame(slave_id: int, address: int, value: int) -> bytes:
    """Modbus RTU FC06 Write Single Register frame."""
    body = struct.pack('>BBHH', slave_id, 0x06, address, value)
    return body + _crc16(body)


def _fc16_frame(slave_id: int, start_address: int, values: list) -> bytes:
    """Modbus RTU FC16 Write Multiple Registers frame."""
    quantity = len(values)
    byte_count = quantity * 2
    header = struct.pack('>BBHHB', slave_id, 0x10, start_address, quantity, byte_count)
    data = b''.join(struct.pack('>H', v) for v in values)
    body = header + data
    return body + _crc16(body)


class DATCGripper:
    CMD_ADDR = 0

    GRIPPER_INITIALIZE = 101
    GRIPPER_OPEN = 102
    GRIPPER_CLOSE = 103
    SET_FINGER_POSITION = 104
    IMPEDANCE_ON = 108
    IMPEDANCE_OFF = 109

    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 38400, slave_id: int = 1):
        self._slave_id = slave_id
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
        print(f"Gripper connected on {port}")
        self.initialize()
        print("Gripper initialized")

    def _send(self, command: int):
        frame = _fc16_frame(self._slave_id, self.CMD_ADDR, [command])
        self._ser.write(frame)

    def initialize(self):
        self._send(self.GRIPPER_INITIALIZE)

    def open(self):
        self._send(self.GRIPPER_OPEN)

    def close(self):
        self._send(self.GRIPPER_CLOSE)

    def set_position(self, position: int):
        """Set finger position. 0=closed, 1000=open."""
        position = max(0, min(1000, int(position)))
        frame = _fc16_frame(self._slave_id, self.CMD_ADDR, [self.SET_FINGER_POSITION, position])
        self._ser.write(frame)

    def impedance_on(self):
        self._send(self.IMPEDANCE_ON)

    def impedance_off(self):
        self._send(self.IMPEDANCE_OFF)

    def _send_fc06(self, command: int):
        """FC06 단일 레지스터 쓰기 (디버그용)."""
        frame = _fc06_frame(self._slave_id, self.CMD_ADDR, command)
        self._ser.write(frame)

    def close_connection(self):
        self._ser.close()
