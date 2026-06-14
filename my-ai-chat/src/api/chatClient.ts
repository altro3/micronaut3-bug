import type { Message } from '../types/chat';

const BASE_URL = 'http://localhost:8083';

export async function sendChatCompletionStream(
    messages: Message[],
    onChunk: (fullCleanText: string) => void
): Promise<void> {
    const response = await fetch(`${BASE_URL}/chat/completions/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            messages: messages.map(({ role, content }) => ({ role, content }))
        }),
    });

    if (!response.ok) {
        throw new Error(`Orchestrator stream error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');

    if (!reader) {
        throw new Error('Response body is not readable');
    }

    let chunkBuffer = '';
    // НАШ НАКОПИТЕЛЬ: Сюда собирается строго чистый неформатированный текст
    let unformattedContent = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunkBuffer += decoder.decode(value, { stream: true });

        // Режем строго по одиночному \n, чтобы разобрать поток на отдельные строки
        const lines = chunkBuffer.split('\n');
        // Остаток строки возвращаем обратно в буфер
        chunkBuffer = lines.pop() || '';

        let hasUpdates = false;

        for (const line of lines) {
            // Очищаем строку от системного Windows-возврата каретки \r, если он есть
            const cleanLine = line.replace(/\r/g, '');

            if (cleanLine.startsWith('data:')) {
                // Отрезаем строго префикс "data:" (первые 5 символов)
                const token = cleanLine.slice(5);

                if (token === '[DONE]') continue;

                if (token === '') {
                    // По спецификации SSE, если после "data:" идет пустота,
                    // значит модель прислала легитимный перенос строки \n
                    unformattedContent += '\n';
                } else {
                    // Во всех остальных случаях приклеиваем токен как есть, со всеми пробелами
                    unformattedContent += token;
                }
                hasUpdates = true;
            }
            // Все пустые строки протокола SSE (разделители блоков) здесь просто ИГНОРИРУЮТСЯ
        }

        if (hasUpdates) {
            // Отдаем ПОЛНЫЙ накопленный чистый текст в UI для переформатирования на каждой итерации
            onChunk(unformattedContent);
        }
    }

    // Выводим финальный лог в консоль СТРОГО ОДИН РАЗ в самом конце
    console.log('%c[FINAL MD CONTENT]:', 'color: #fbbf24; font-weight: bold;\n', unformattedContent);
}
