"""
Реализация транспортного протокола SerProt v0.2.

Структура фрейма:
    0xAA | type | seq | len_L | len_H | payload | CRC16_L | CRC16_H

Escape-кодирование применяется только к payload и CRC-16.
CRC-16/Modbus покрывает [type][seq][len_L][len_H][payload].
"""

from dataclasses import dataclass
from typing import Optional


# --- CRC-16/Modbus ------------------------------------------------------------

def crc16_modbus(data: bytes) -> int:
    """CRC-16/Modbus: полином 0x8005 (reflected 0xA001), init=0xFFFF."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _crc16_update(crc: int, byte: int) -> int:
    """Побайтовое обновление CRC-16/Modbus."""
    crc ^= byte
    for _ in range(8):
        if crc & 0x0001:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc


# --- Frame -------------------------------------------------------------------

@dataclass(frozen=True)
class SerProtFrame:
    type: int
    seq: int
    payload: bytes


# --- Parser ------------------------------------------------------------------

class SerProtParser:
    """Конечный автомат разбора SerProt v0.2 (по Machine_state_parser.csv)."""

    # Состояния
    S0_WAIT_START = 0
    S1_TYPE = 1
    S2_SEQ = 2
    S3_LEN_L = 3
    S4_LEN_H = 4
    S5_PAYLOAD = 5
    S6_CRC_L = 6
    S7_CRC_H = 7
    S_ESC_PAYLOAD = 8
    S_ESC_CRC_L = 9
    S_ESC_CRC_H = 10

    def __init__(self):
        self._state = self.S0_WAIT_START
        self._type = 0
        self._seq = 0
        self._payload_len = 0
        self._payload_cnt = 0
        self._payload_buf = bytearray()
        self._crc_l = 0
        self._crc_h = 0

    def reset(self):
        """Сбросить парсер в начальное состояние."""
        self._state = self.S0_WAIT_START
        self._type = 0
        self._seq = 0
        self._payload_len = 0
        self._payload_cnt = 0
        self._payload_buf.clear()
        self._crc_l = 0
        self._crc_h = 0

    def process_byte(self, b: int) -> Optional[SerProtFrame]:
        """Обработать один байт. Вернуть frame, если пакет собран и CRC OK."""
        s = self._state

        if s == self.S0_WAIT_START:
            if b == 0xAA:
                self._state = self.S1_TYPE

        elif s == self.S1_TYPE:
            self._type = b
            self._state = self.S2_SEQ

        elif s == self.S2_SEQ:
            self._seq = b
            self._state = self.S3_LEN_L

        elif s == self.S3_LEN_L:
            self._payload_len = b
            self._state = self.S4_LEN_H

        elif s == self.S4_LEN_H:
            self._payload_len |= (b << 8)
            self._payload_cnt = self._payload_len
            self._payload_buf.clear()
            if self._payload_len == 0:
                self._state = self.S6_CRC_L
            else:
                self._state = self.S5_PAYLOAD

        elif s == self.S5_PAYLOAD:
            if b == 0xBB:
                self._state = self.S_ESC_PAYLOAD
            else:
                self._payload_buf.append(b)
                self._payload_cnt -= 1
                if self._payload_cnt == 0:
                    self._state = self.S6_CRC_L

        elif s == self.S_ESC_PAYLOAD:
            if b == 0xAA or b == 0xBB:
                self._payload_buf.append(b)
                self._payload_cnt -= 1
                if self._payload_cnt == 0:
                    self._state = self.S6_CRC_L
                else:
                    self._state = self.S5_PAYLOAD
            else:
                self._state = self.S0_WAIT_START

        elif s == self.S6_CRC_L:
            if b == 0xBB:
                self._state = self.S_ESC_CRC_L
            else:
                self._crc_l = b
                self._state = self.S7_CRC_H

        elif s == self.S7_CRC_H:
            if b == 0xBB:
                self._state = self.S_ESC_CRC_H
            else:
                self._crc_h = b
                return self._verify_and_reset()

        elif s == self.S_ESC_CRC_L:
            if b == 0xAA or b == 0xBB:
                self._crc_l = b
                self._state = self.S7_CRC_H
            else:
                self._state = self.S0_WAIT_START

        elif s == self.S_ESC_CRC_H:
            if b == 0xAA or b == 0xBB:
                self._crc_h = b
                return self._verify_and_reset()
            else:
                self._state = self.S0_WAIT_START

        return None

    def _verify_and_reset(self) -> Optional[SerProtFrame]:
        """Проверить CRC и вернуть frame. В любом случае сбросить состояние."""
        crc_computed = 0xFFFF
        crc_computed = _crc16_update(crc_computed, self._type)
        crc_computed = _crc16_update(crc_computed, self._seq)
        crc_computed = _crc16_update(crc_computed, self._payload_len & 0xFF)
        crc_computed = _crc16_update(crc_computed, (self._payload_len >> 8) & 0xFF)
        for b in self._payload_buf:
            crc_computed = _crc16_update(crc_computed, b)

        crc_received = self._crc_l | (self._crc_h << 8)
        self._state = self.S0_WAIT_START

        if crc_computed == crc_received:
            return SerProtFrame(
                type=self._type,
                seq=self._seq,
                payload=bytes(self._payload_buf)
            )
        else:
            print(f"[RX] ERR CRC mismatch type=0x{self._type:02X} seq={self._seq:02X} "
                  f"computed=0x{crc_computed:04X} received=0x{crc_received:04X} "
                  f"payload_len={self._payload_len} payload={self._payload_buf.hex()}")
            return None


# --- Builder / Master --------------------------------------------------------

class SerProtMaster:
    """Высокоуровневый интерфейс для формирования SerProt-фреймов."""

    TYPE_COMMAND = 0xCC
    TYPE_RESPONSE = 0xDD
    TYPE_ERROR = 0xEE
    TYPE_PUSH = 0xFF

    def __init__(self):
        self._seq = 0

    def next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return s

    def build_command(self, cmd_id: int, payload: bytes = b"") -> bytes:
        """Собрать Command-фрейм (0xCC)."""
        full_payload = bytes([cmd_id]) + payload
        return self.build_frame(self.TYPE_COMMAND, self.next_seq(), full_payload)

    @staticmethod
    def build_frame(type_byte: int, seq: int, payload: bytes) -> bytes:
        """Собрать фрейм SerProt v0.2 с escape-кодированием payload и CRC."""
        length = len(payload)
        # CRC вычисляется над неэкранированными данными
        crc_data = bytes([type_byte, seq, length & 0xFF, (length >> 8) & 0xFF]) + payload
        crc = crc16_modbus(crc_data)
        crc_l = crc & 0xFF
        crc_h = (crc >> 8) & 0xFF

        # Собираем фрейм
        frame = bytearray()
        frame.append(0xAA)
        frame.append(type_byte)
        frame.append(seq)
        frame.append(length & 0xFF)
        frame.append((length >> 8) & 0xFF)

        # Escape payload
        for b in payload:
            if b == 0xAA or b == 0xBB:
                frame.append(0xBB)
            frame.append(b)

        # Escape CRC
        for b in (crc_l, crc_h):
            if b == 0xAA or b == 0xBB:
                frame.append(0xBB)
            frame.append(b)

        return bytes(frame)

    @staticmethod
    def decode_adc_samples(payload: bytes) -> list[float]:
        """Декодировать payload Push-пакета как 24-bit signed LE → мкВ (float)."""
        if len(payload) % 3 != 0:
            raise ValueError(f"Payload length {len(payload)} is not divisible by 3")
        samples = []
        for i in range(0, len(payload), 3):
            raw = payload[i] | (payload[i + 1] << 8) | (payload[i + 2] << 16)
            if raw >= 0x800000:
                raw -= 0x1000000
            samples.append(float(raw))
        return samples
