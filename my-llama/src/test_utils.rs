pub fn emu_fp4_e2m1_to_f32(byte: u8, idx: usize) -> f32 {
    let nibble = if idx % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
    let s = (nibble >> 3) & 1;
    let e = (nibble >> 1) & 3;
    let m = nibble & 1;

    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        if m == 0 {
            return 0.0;
        }
        return sign * 1.0 * (m as f32 / 2.0);
    }
    let exp = e as i32 - 1;
    let mantissa = 1.0f32 + (m as f32 / 2.0f32);
    sign * mantissa * 2.0f32.powi(exp)
}

pub fn emu_fp8_e4m3_to_f32(byte: u8) -> f32 {
    if byte == 0x7F || byte == 0xFF {
        return f32::NAN;
    }

    let s = (byte >> 7) & 1;
    let e = (byte >> 3) & 0x0F;
    let m = byte & 7;

    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        if m == 0 {
            return 0.0;
        }
        return sign * 2.0f32.powi(-6) * (m as f32 / 8.0);
    }

    let exp = e as i32 - 7;
    let mantissa = 1.0f32 + (m as f32 / 8.0f32);
    sign * mantissa * 2.0f32.powi(exp)
}

pub fn f32_to_bf16_bits(val: f32) -> u16 {
    if val.is_nan() {
        return 0x7FC0;
    }
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded_bits = bits.wrapping_add(rounding_bias);
    ((rounded_bits >> 16) & 0xFFFF) as u16
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
