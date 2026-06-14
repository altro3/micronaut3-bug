import type {Message} from '../types/chat';

const BASE_URL = 'http://localhost:8083';

export async function sendChatCompletion(messages: Message[]): Promise<string> {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'qwen-3.6',
            messages: messages.map(({role, content}) => ({role, content}))
        }),
    });

    if (!response.ok) {
        throw new Error(`Orchestrator responded with status: ${response.status}`);
    }

    const data = await response.json();

    // Безопасно извлекаем контент по спецификации OpenAI
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
        throw new Error('Empty content in OpenAI response structure');
    }

    return content;
}
