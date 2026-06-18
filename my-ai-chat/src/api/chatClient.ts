import type {Message} from '../types/chat';

const BASE_URL = 'http://localhost:8083';

export async function sendChatCompletionStream(
    messages: Message[],
    sessionId: string | null,
    onChunk: (fullCleanText: string) => void,
    onSessionId: (id: string) => void
): Promise<void> {
    const response = await fetch(`${BASE_URL}/chat/completions/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
            sessionId: sessionId,
            messages: messages.map(({role, content}) => ({role, content}))
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
    let unformattedContent = '';
    let sessionCaptured = false;

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        chunkBuffer += decoder.decode(value, {stream: true});

        const lines = chunkBuffer.split('\n');
        chunkBuffer = lines.pop() || '';

        let hasUpdates = false;

        for (const line of lines) {
            const cleanLine = line.replace(/\r/g, '');

            if (cleanLine === '') continue;

            if (cleanLine.startsWith('data:')) {
                let token = cleanLine.slice(5);

                if (token === '[DONE]') continue;

                // 1. ПЕРЕХВАТ ТЕХНИЧЕСКОГО МАРКЕРА СЕССИИ
                if (!sessionCaptured && token.includes('[SESSION_ID:')) {
                    const match = token.match(/\[SESSION_ID:([a-f0-9-]+)]/i);
                    if (match && match[1]) {
                        const pureUuid = match[1];
                        console.log("✈️ Перехвачен UUID сессии с бэка:", pureUuid);
                        onSessionId(pureUuid);
                        sessionCaptured = true;
                    }
                    token = token.replace(/\[SESSION_ID:.+?]/, '');
                }

                // 2. ЖЕСТКИЙ РАЗДЕЛИТЕЛЬ ДЛЯ MD-ФОРМАТА (ДВЕ СТРОКИ)
                if (token === '') {
                    // Если пришла пустая строка data:, гарантируем честный пустой абзац (\n\n)
                    if (!unformattedContent.endsWith('\n\n')) {
                        unformattedContent = unformattedContent.replace(/\n*$/, '') + '\n\n';
                    }
                } else {
                    // Если это новый элемент списка, заголовок или маркер, форсируем ДВА переноса перед ним
                    const trimToken = token.trim();
                    const isNewBlock = trimToken.startsWith('•') ||
                        trimToken.startsWith('*') ||
                        /^\d+\./.test(trimToken); // Проверка на "1.", "2." и т.д.

                    if (unformattedContent.length > 0) {
                        if (isNewBlock) {
                            // Перед заголовками и списками всегда бахаем двойной перенос
                            if (!unformattedContent.endsWith('\n\n')) {
                                unformattedContent = unformattedContent.replace(/\n*$/, '') + '\n\n';
                            }
                        } else if (!unformattedContent.endsWith('\n')) {
                            // Перед обычным текстом — один
                            unformattedContent += '\n';
                        }
                    }
                    unformattedContent += token;
                }
                hasUpdates = true;
            } else {
                // Фолбек для сырых строк без data:
                if (unformattedContent.length > 0 && !unformattedContent.endsWith('\n')) {
                    unformattedContent += '\n';
                }
                unformattedContent += cleanLine;
                hasUpdates = true;
            }
        }

        if (hasUpdates) {
            // Безопасная очистка Markdown-обертки без удаления внутренних \n
            let cleanedContent = unformattedContent;

            if (cleanedContent.startsWith('```markdown')) {
                cleanedContent = cleanedContent.slice(11);
            } else if (cleanedContent.startsWith('```')) {
                cleanedContent = cleanedContent.slice(3);
            }

            if (cleanedContent.endsWith('```')) {
                cleanedContent = cleanedContent.slice(0, -3);
            }

            onChunk(cleanedContent);
        }
    }

    console.log('%c[FINAL MD CONTENT]:', 'color: #fbbf24; font-weight: bold;\n', unformattedContent);
}
