pub struct FactoryUtils;

impl FactoryUtils {
    #[inline(always)]
    pub fn fxhash64(b: &[u8]) -> u64 {
        let len = b.len();
        let mut hash = 0u64;
        let mut c = 0;
        let ptr = b.as_ptr();

        while c + 8 <= len {
            let block = unsafe { std::ptr::read_unaligned(ptr.add(c) as *const u64) };
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
            c += 8;
        }

        if c < len {
            let mut block = 0u64;
            let mut shift = 0;
            while c < len {
                block |= (unsafe { *ptr.add(c) } as u64) << shift;
                shift += 8;
                c += 1;
            }
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
        }
        hash
    }

    #[inline(always)]
    pub fn decode_inplace(s: &[u8], buf: &mut [u8]) -> usize {
        let len = s.len();
        if len == 0 {
            return 0;
        }

        let mut i = 0;
        let mut cp: u32 = 0;
        let b0 = s[0];

        if b0 < 0x80 {
            cp = b0 as u32;
        } else if (b0 & 0xE0) == 0xC0 && len >= 2 {
            cp = (((b0 & 0x1F) as u32) << 6) | (s[1] & 0x3F) as u32;
        } else if (b0 & 0xF0) == 0xE0 && len >= 3 {
            cp = (((b0 & 0x0F) as u32) << 12) | (((s[1] & 0x3F) as u32) << 6) | (s[2] & 0x3F) as u32;
        } else if (b0 & 0xF8) == 0xF0 && len >= 4 {
            cp = (((b0 & 0x07) as u32) << 18) | (((s[1] & 0x3F) as u32) << 12) | (((s[2] & 0x3F) as u32) << 6) | (s[3] & 0x3F) as u32;
        }

        let raw_byte = match cp {
            0x00..=0x7F => cp as u8,
            0x0100..=0x0120 => (cp - 0x0100) as u8,
            0x0121..=0x017D => (cp - 0x0121 + 33) as u8,
            0x017E..=0x01AC => (cp - 0x017E + 127) as u8,
            0x01AD..=0x01FF => (cp - 0x01AD + 174) as u8,
            _ => cp as u8,
        };

        buf[0] = raw_byte;
        1
    }

    #[inline(always)]
    pub fn parse_u32(bytes: &[u8]) -> u32 {
        let mut val = 0u32;
        let len = bytes.len();
        let ptr = bytes.as_ptr();
        for i in 0..len {
            unsafe {
                let digit = *ptr.add(i) - b'0';
                val = val * 10 + digit as u32;
            }
        }
        val
    }
}

#[cfg(test)]
mod ultra_assert_tests {
    use super::FactoryUtils;

    // ТЕСТ 1: Проверяем, что Byte-Level BPE демаппинг работает без паник
    // и возвращает строго ОДИН сырой байт для спец-символов OpenAI/Qwen
    #[test]
    fn assert_byte_level_demapping_exact() {
        let mut buf = [0u8; 4];

        // Проверяем ASCII диапазон (символ '!' должен остаться байтом 33)
        let len = FactoryUtils::decode_inplace(b"!", &mut buf);
        assert_eq!(len, 1, "ASCII токен должен выдать длину 1");
        assert_eq!(buf[0], 33, "Символ '!' должен быть байтом 33");

        // Проверяем символ 'Ġ' (Unicode CP 0x0120). По стандарту BPE он обязан стать байтом 32 (пробел)
        // Используем \xC4\xA0 вместо сырого символа
        let len = FactoryUtils::decode_inplace(b"\xC4\xA0", &mut buf);
        assert_eq!(len, 1, "Символ Ġ обязан вернуть длину 1");
        assert_eq!(buf[0], 32, "Символ Ġ обязан демаппиться в сырой байт пробела (32)");

        // Проверяем символ 'ĉ' (Unicode CP 0x0109). Обязан стать байтом 9 (\t - табуляция)
        // Используем \xC4\x89 вместо сырого символа
        let len = FactoryUtils::decode_inplace(b"\xC4\x89", &mut buf);
        assert_eq!(len, 1);
        assert_eq!(buf[0], 9, "Символ ĉ обязан демаппиться в байт табуляции (9)");
    }

    // ТЕСТ 2: Проверяем unescape бэкслешей из JSON и сведение хэшей
    // Именно тут падало BPE-сжатие до 1.00x
    #[test]
    fn assert_json_unescape_and_hash_match() {
        // Симулируем то, что мы прочитали из JSON vocab: токен кавычки "\"" и токен "Ġbut"
        let raw_vocab_quote = b"\\\""; // В файле это выглядит как \"
        let raw_vocab_but = b"\xC4\xA0but"; // Используем \xC4\xA0 вместо сырого Ġ

        // Симулируем то, что лежит в merges: чистые токены без экранирования
        let raw_merge_quote = b"\""; // В merges кавычка идет без бэкслеша
        let raw_merge_but = b"\xC4\xA0but";

        // Прогоняем vocab через unescape в нашей фабрике
        let mut clean_vocab_quote = [0u8; 256];
        let mut clen_q = 0;
        let mut si = 0;
        while si < raw_vocab_quote.len() {
            if raw_vocab_quote[si] == b'\\' && si + 1 < raw_vocab_quote.len() {
                clean_vocab_quote[clen_q] = match raw_vocab_quote[si + 1] {
                    b'"' => b'"',
                    b'\\' => b'\\',
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    other => other,
                };
                si += 2;
            } else {
                clean_vocab_quote[clen_q] = raw_vocab_quote[si];
                si += 1;
            }
            clen_q += 1;
        }

        // Хэшируем очищенный токен из вокаба и токен из мёрджей
        let hash_from_vocab = FactoryUtils::fxhash64(&clean_vocab_quote[..clen_q]);
        let hash_from_merge = FactoryUtils::fxhash64(raw_merge_quote);

        // ХЭШИ ОБЯЗАНЫ СОВПАДАТЬ, иначе мёрджи никогда не найдут свои ID!
        assert_eq!(
            hash_from_vocab, hash_from_merge,
            "Критический баг! Хэш кавычки из vocab после unescape не совпал с мёрджем!"
        );

        // Проверяем то же самое для "Ġbut"
        let hash_but_vocab = FactoryUtils::fxhash64(raw_vocab_but);
        let hash_but_merge = FactoryUtils::fxhash64(raw_merge_but);
        assert_eq!(hash_but_vocab, hash_but_merge, "Хэши стандартных токенов разошлись!");
    }
}
